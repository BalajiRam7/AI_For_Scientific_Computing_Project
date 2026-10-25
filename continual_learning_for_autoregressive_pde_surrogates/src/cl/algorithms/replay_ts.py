import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

from .algorithm import Algorithm
from .utils.memory import Memory as BaseMemory
from src.common.stream import Experience
from src.common.logger import Logger


class ReplayTS(Algorithm):
    def __init__(
        self,
        model: nn.Module,
        device: torch.DeviceObjType = torch.device("cpu"),
        num_workers: int = 0,
        logger: Logger = None,
        optimizer: str = "adam",
        optimizer_params: dict = {},
        criterion: str = "mse",
        criterion_params: dict = {},
        batch_size: int = 64,
        epochs: int = 5,
        memory_size: int = 32,
        memory_seed: int = 42,
        memory_mode: str = "reservoir",
        replay_loss_coef: float = 1.0,
        replay_batch_size: int = 32,
        replay_slice_factor: int = 1,
        log_grad_conflict: bool = False,
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

        self.replay_loss_coef = replay_loss_coef
        self.replay_batch_size = replay_batch_size
        self.replay_slice_factor = replay_slice_factor
        self.memory = SlicedMemory(
            memory_size=memory_size,
            device=self.device,
            memory_seed=memory_seed,
            slice_factor=replay_slice_factor,
            mode=memory_mode,
        )
        self.log_grad_conflict = log_grad_conflict

    def calculate_grad_conflict(self, loss_a, loss_b):
        # Compute gradients safely (skip params with no grad connection)
        params = [p for p in self.model.parameters() if p.requires_grad]

        grads_a = torch.autograd.grad(
            loss_a, params, retain_graph=True, allow_unused=True
        )
        grads_b = torch.autograd.grad(
            loss_b, params, retain_graph=True, allow_unused=True
        )

        flat_a, flat_b = [], []
        for g1, g2 in zip(grads_a, grads_b):
            if g1 is None or g2 is None:
                continue
            if torch.is_complex(g1):
                g1 = torch.view_as_real(g1)
                g2 = torch.view_as_real(g2)
            flat_a.append(g1.reshape(-1))
            flat_b.append(g2.reshape(-1))

        if not flat_a or not flat_b:
            return 0.0  # no valid gradients

        flat_a = torch.cat(flat_a)
        flat_b = torch.cat(flat_b)

        cos_sim = torch.dot(flat_a, flat_b) / (flat_a.norm() * flat_b.norm() + 1e-8)
        return cos_sim.item()

    def train_exp(self, experience: Experience):
        self.model.train()
        train_loader = DataLoader(
            experience.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )

        for epoch in range(self.epochs):
            train_progress = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True
            )
            for batch in train_progress:
                # Loss for the current batch
                x, y = batch["input"].to(self.device), batch["output"].to(self.device)
                n_rollout_steps = y.shape[1]
                self.optimizer.zero_grad()
                total_loss = 0.0
                pred_loss = 0.0
                for i in range(n_rollout_steps):
                    pred_i = self.model(x, aux_input=batch["aux_input"])
                    pred_loss += self.criterion(pred_i, y[:, i, :].unsqueeze(1))
                    x = torch.concat([x, pred_i], dim=1)[:, 1:, :]
                total_loss += pred_loss

                # Replay loss
                if len(self.memory) > 0:
                    x_mem, y_mem, aux_input_mem = self.memory.sample_from_memory(
                        num_samples=self.replay_batch_size
                    )
                    mem_loss = 0.0
                    for i in range(n_rollout_steps):
                        pred_i = self.model(x_mem, aux_input=aux_input_mem)
                        if i % self.replay_slice_factor == 0:
                            k = i // self.replay_slice_factor
                            mem_loss += self.criterion(
                                pred_i, y_mem[:, k, :].unsqueeze(1)
                            )
                        x_mem = torch.concat([x_mem, pred_i], dim=1)[:, 1:, :]
                    total_loss += self.replay_loss_coef * mem_loss

                # Grad conflict
                if self.log_grad_conflict:
                    grad_conflict = self.calculate_grad_conflict(pred_loss, mem_loss)
                    self.log_metric(
                        phase="train", key="grad_conflict", value=grad_conflict
                    )

                total_loss.backward()
                self.optimizer.step()

                # Logging
                self.log_metric(phase="train", key="step_loss", value=total_loss.item())

                if len(self.memory) > 0:
                    self.log_metric(
                        phase="train", key="pred_loss", value=pred_loss.item()
                    )
                    self.log_metric(
                        phase="train", key="mem_loss", value=mem_loss.item()
                    )
                self.global_step += 1

        self.log_metric(phase="train", key="exp_idx", value=experience.exp_idx)

        # After each experience, update memory with a pass over the experience data
        self.memory.add_samples_from_experience(experience, batch_size=self.batch_size)

    def save_checkpoint_after_exp(self, experience: Experience):
        checkpoint_path = os.path.join(
            self.logger.checkpoints_dir, f"model_weights_e{experience.exp_idx}.pt"
        )
        torch.save(self.model.state_dict(), checkpoint_path)

        # Also save the optimizer state
        optimizer_path = os.path.join(
            self.logger.checkpoints_dir, f"optim_state_e{experience.exp_idx}.pt"
        )
        torch.save(self.optimizer.state_dict(), optimizer_path)


class SlicedMemory(BaseMemory):
    def __init__(
        self,
        memory_size,
        device,
        memory_seed: int,
        slice_factor: int = 1,
        mode="reservoir",
    ):
        super().__init__(
            memory_size=memory_size,
            device=device,
            memory_seed=memory_seed,
            mode=mode,
        )
        self.slice_factor = max(1, int(slice_factor))

    def sample_from_memory(self, num_samples: int):
        xs, ys, aux = super().sample_from_memory(num_samples)
        if ys.numel() == 0 or self.slice_factor == 1:
            return xs, ys, aux
        ys = ys[:, :: self.slice_factor, :]
        return xs, ys, aux
