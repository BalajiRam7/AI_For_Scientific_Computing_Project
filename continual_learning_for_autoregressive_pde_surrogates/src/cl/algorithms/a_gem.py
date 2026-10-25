import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .algorithm import Algorithm
from .utils.memory import Memory
from src.common.stream import Experience
from src.common.logger import Logger


class AGEM(Algorithm):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 0,
        logger: Logger = None,
        optimizer: str = "adam",
        optimizer_params: dict = {},
        criterion: str = "mse",
        criterion_params: dict = {},
        batch_size: int = 64,
        epochs: int = 5,
        memory_size: int = 1000,
        memory_batch_size: int = 32,
        memory_mode: str = "experience_balanced",
        eps_agem: float = 1e-7,
    ):
        super().__init__(
            model,
            device,
            logger=logger,
            num_workers=num_workers,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            criterion=criterion,
            criterion_params=criterion_params,
            batch_size=batch_size,
            epochs=epochs,
        )

        self.eps_agem = eps_agem
        self.memory_batch_size = memory_batch_size
        self.memory = Memory(memory_size=memory_size, device=device, mode=memory_mode)

        # Fixed-layout gradient vectors (aligned with GEM)
        self.grad_dims = [p.data.numel() for p in self.model.parameters()]
        self.grads_da = torch.zeros(int(np.sum(self.grad_dims)), device=self.device)

    def _store_grad(self):
        """
        Flatten all parameter gradients into one real vector.
        Complex grads are concatenated as [Re, Im].
        """
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach().clone().flatten()
                if torch.is_complex(g):
                    g = torch.cat([g.real, g.imag])
                grads.append(g)
        return torch.cat(grads)

    def _store_grad_aligned(self, dest_vec: torch.Tensor):
        """Store current grads into dest_vec using fixed grad_dims layout."""
        dest_vec.zero_()
        count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                begin = 0 if count == 0 else int(np.sum(self.grad_dims[:count]))
                end = int(np.sum(self.grad_dims[: count + 1]))
                dest_vec[begin:end].copy_(param.grad.view(-1))
            count += 1

    def _overwrite_grad(self, new_grad):
        """
        Overwrite model gradients with flattened vector.
        Writes only for params that had gradients in the stored vector,
        to stay aligned with _store_grad slicing.
        """
        start = 0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            numel = p.numel()
            if torch.is_complex(p):
                real_slice = new_grad[start : start + numel]
                imag_slice = new_grad[start + numel : start + 2 * numel]
                g_slice = torch.complex(real_slice, imag_slice)
                start += 2 * numel
            else:
                g_slice = new_grad[start : start + numel]
                start += numel

            p.grad.detach_()
            p.grad.copy_(g_slice.reshape_as(p))
        return

    def _overwrite_grad_aligned(self, src_vec: torch.Tensor):
        """Overwrite model grads from src_vec using fixed grad_dims layout.
        Allocates param.grad if needed to be robust to zero_grad(set_to_none=True).
        """
        count = 0
        for param in self.model.parameters():
            begin = 0 if count == 0 else int(np.sum(self.grad_dims[:count]))
            end = int(np.sum(self.grad_dims[: count + 1]))
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            this_grad = src_vec[begin:end].contiguous().view_as(param.grad)
            param.grad.copy_(this_grad)
            count += 1

    def train_exp(self, experience: Experience):
        self.model.train()
        train_loader = DataLoader(
            experience.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        for epoch in range(self.epochs):
            train_progress = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True
            )
            for batch in train_progress:
                x, y = batch["input"].to(self.device), batch["output"].to(self.device)
                aux_input = batch["aux_input"].to(self.device)
                n_rollout_steps = y.shape[1]

                # ---- Compute current gradient (autoregressive loss) ----
                self.optimizer.zero_grad()
                total_loss = 0.0
                x_cur = x.clone()

                for i in range(n_rollout_steps):
                    pred_i = self.model(x_cur, aux_input=aux_input)
                    total_loss += self.criterion(pred_i, y[:, i, :].unsqueeze(1))
                    x_cur = torch.cat([x_cur, pred_i], dim=1)[:, 1:, :]

                total_loss.backward()
                self._store_grad_aligned(self.grads_da)

                # ---- A-GEM projection using memory gradients ----
                if len(self.memory) > 0:
                    x_mem, y_mem, aux_input_mem = self.memory.sample_from_memory(
                        num_samples=self.memory_batch_size
                    )

                    self.optimizer.zero_grad()
                    mem_loss = 0.0
                    x_mem_cur = x_mem.clone()

                    for i in range(n_rollout_steps):
                        pred_i_mem = self.model(x_mem_cur, aux_input=aux_input_mem)
                        mem_loss += self.criterion(
                            pred_i_mem, y_mem[:, i, :].unsqueeze(1)
                        )
                        x_mem_cur = torch.cat([x_mem_cur, pred_i_mem], dim=1)[:, 1:, :]

                    mem_loss.backward()
                    mem_grad = torch.zeros_like(self.grads_da)
                    self._store_grad_aligned(mem_grad)

                    # ---- Projection step ----
                    dotp = torch.dot(self.grads_da, mem_grad)
                    if dotp < 0:
                        mem_norm = torch.dot(mem_grad, mem_grad)
                        proj_coef = dotp / (mem_norm + self.eps_agem)
                        proj_grad = self.grads_da - proj_coef * mem_grad
                        self.optimizer.zero_grad()
                        self._overwrite_grad_aligned(proj_grad)
                    else:
                        self.optimizer.zero_grad()
                        self._overwrite_grad_aligned(self.grads_da)
                else:
                    # No memory yet
                    self.optimizer.zero_grad()
                    self._overwrite_grad_aligned(self.grads_da)

                self.optimizer.step()

                # ---- Logging ----
                self.log_metric(phase="train", key="step_loss", value=total_loss.item())
                self.global_step += 1

        self.log_metric(phase="train", key="exp_idx", value=experience.exp_idx)
        self.memory.add_samples_from_experience(experience, batch_size=self.batch_size)
