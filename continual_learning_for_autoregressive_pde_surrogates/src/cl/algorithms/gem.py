import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .algorithm import Algorithm
from .utils.memory import Memory
from src.common.stream import Experience
from src.common.logger import Logger


class GEM(Algorithm):
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
        eps_gem: float = 1e-7,
        gamma: float = 0.5,
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

        self.eps_gem = eps_gem
        self.gamma = gamma
        self.memory_batch_size = memory_batch_size
        self.memory = Memory(memory_size=memory_size, device=device, mode=memory_mode)

        # --- GEM state aligned with Mammoth ---
        self.grad_dims = [p.data.numel() for p in self.model.parameters()]
        self.grads_da = torch.zeros(int(np.sum(self.grad_dims)), device=self.device)
        self.grads_cs = []  # list of per-exp gradient vectors

        # Try to import quadprog solver like Mammoth (optional)
        self._gem_solver = None
        try:
            import quadprog as _solver  # type: ignore

            self._gem_solver = _solver
        except Exception:
            self._gem_solver = None

    def _store_grad(self):
        """
        Flatten all parameter gradients into a single real-valued vector.
        Complex gradients are split into [Re, Im].
        """
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach().clone().flatten()
                if torch.is_complex(g):
                    g = torch.cat([g.real, g.imag])
                grads.append(g)
        return torch.cat(grads)

    def _overwrite_grad(self, new_grad):
        """
        Overwrite model gradients with a new flattened vector.
        Only writes to parameters that had gradients in the stored vector.
        This keeps slicing consistent with _store_grad (which skips None grads).
        """
        start = 0
        for p in self.model.parameters():
            if p.grad is None:
                # this parameter did not contribute to the stored gradient
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

    def _project_grad(self, current_grad, mem_grads):
        """
        GEM projection step ensuring ⟨g_t, g_j⟩ ≥ 0 for all j.
        Operates on real-valued flattened gradients.
        """
        projected_grad = current_grad.clone()
        for g_mem in mem_grads:
            dotp = torch.dot(projected_grad, g_mem)
            if dotp < 0:
                mem_norm_sq = torch.dot(g_mem, g_mem)
                proj_coef = dotp / (mem_norm_sq + self.eps_gem)
                projected_grad -= proj_coef * g_mem
        return projected_grad

    # ===== Methods aligned with Mammoth GEM behavior =====
    def _ensure_exp_slot(self, exp_id: int):
        while len(self.grads_cs) <= int(exp_id):
            self.grads_cs.append(
                torch.zeros(int(np.sum(self.grad_dims)), device=self.device)
            )

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

    def _overwrite_grad_aligned(self, src_vec: torch.Tensor):
        """Overwrite model grads from src_vec using fixed grad_dims layout.
        Allocates param.grad if needed to be robust to zero_grad(set_to_none=True).
        """
        count = 0
        for param in self.model.parameters():
            begin = 0 if count == 0 else int(np.sum(self.grad_dims[:count]))
            end = int(np.sum(self.grad_dims[: count + 1]))
            # Ensure grad exists
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            # Copy slice
            this_grad = src_vec[begin:end].contiguous().view_as(param.grad)
            param.grad.copy_(this_grad)
            count += 1

    def _project2cone_qp(self, gradient: torch.Tensor, memories: torch.Tensor):
        """QP-based projection like Mammoth using quadprog if available.
        gradient: (p,)
        memories: (t, p)
        """
        if self._gem_solver is None:
            return None
        memories_np = memories.detach().cpu().double().numpy()
        gradient_np = gradient.detach().cpu().contiguous().view(-1).double().numpy()
        n_rows = memories_np.shape[0]
        self_prod = np.dot(memories_np, memories_np.T)
        self_prod = 0.5 * (self_prod + self_prod.T) + np.eye(n_rows) * 1e-3
        grad_prod = np.dot(memories_np, gradient_np) * -1
        G = np.eye(n_rows)
        h = np.zeros(n_rows) + self.gamma
        try:
            v = self._gem_solver.solve_qp(self_prod, grad_prod, G, h)[0]
        except Exception:
            return None
        x = np.dot(v, memories_np) + gradient_np
        out = torch.from_numpy(x).to(gradient.device, dtype=gradient.dtype)
        gradient.copy_(out)
        return gradient

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
                aux_raw = batch.get("aux_input")
                aux_input = (
                    aux_raw.to(self.device) if torch.is_tensor(aux_raw) else None
                )
                n_rollout_steps = y.shape[1]

                if len(self.memory) > 0:
                    x_mem, y_mem, aux_input_mem, exp_labels = (
                        self.memory.sample_from_memory(
                            num_samples=self.memory_batch_size, return_exp_ids=True
                        )
                    )

                    for tt in exp_labels.unique():
                        idx = exp_labels == tt
                        if idx.sum() == 0:
                            continue
                        self.optimizer.zero_grad()
                        x_tt = x_mem[idx]
                        y_tt = y_mem[idx]
                        aux_tt = (
                            aux_input_mem[idx] if aux_input_mem.numel() > 0 else None
                        )

                        mem_loss = 0.0
                        x_mem_cur = x_tt.clone()
                        for i in range(n_rollout_steps):
                            pred_i_mem = self.model(x_mem_cur, aux_input=aux_tt)
                            mem_loss += self.criterion(
                                pred_i_mem, y_tt[:, i, :].unsqueeze(1)
                            )
                            x_mem_cur = torch.cat([x_mem_cur, pred_i_mem], dim=1)[
                                :, 1:, :
                            ]
                        mem_loss.backward()
                        # store into per-exp gradient slot
                        self._ensure_exp_slot(int(tt.item()))
                        self._store_grad_aligned(self.grads_cs[int(tt.item())])

                # --- Current gradient on present batch ---
                self.optimizer.zero_grad()
                total_loss = 0.0
                x_cur = x.clone()
                for i in range(n_rollout_steps):
                    pred_i = self.model(x_cur, aux_input=aux_input)
                    total_loss += self.criterion(pred_i, y[:, i, :].unsqueeze(1))
                    x_cur = torch.cat([x_cur, pred_i], dim=1)[:, 1:, :]
                total_loss.backward()

                # Copy current flatten grad
                self._store_grad_aligned(self.grads_da)

                # Constraint check and projection
                need_project = False
                if len(self.grads_cs) > 0:
                    mem_mat = torch.stack(self.grads_cs)  # shape (num_exps, p)
                    dotp = torch.mm(self.grads_da.unsqueeze(0), mem_mat.T)
                    need_project = (dotp < 0).any().item()

                if need_project:
                    # Try QP first
                    qp_out = self._project2cone_qp(self.grads_da, mem_mat)
                    if qp_out is None:
                        projected = self.grads_da.clone()
                        for g_mem in mem_mat:
                            mem_norm_sq = torch.dot(g_mem, g_mem)
                            if mem_norm_sq <= self.eps_gem:
                                continue
                            dp = torch.dot(projected, g_mem)
                            if dp < 0:
                                projected -= (dp / mem_norm_sq) * g_mem
                        self.grads_da.copy_(projected)

                    # Overwrite projected grads back to model and step
                    self.optimizer.zero_grad()
                    self._overwrite_grad_aligned(self.grads_da)
                    self.optimizer.step()
                else:
                    self.optimizer.step()

                self.log_metric(phase="train", key="step_loss", value=total_loss.item())
                self.global_step += 1

        self.log_metric(phase="train", key="exp_idx", value=experience.exp_idx)
        self.memory.add_samples_from_experience(experience, batch_size=self.batch_size)
        # allocate a new exp slot at end of experience (like Mammoth end_exp)
        self._ensure_exp_slot(experience.exp_idx)
