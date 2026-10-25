from typing import List, Tuple

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .algorithm import Algorithm
from .utils.memory import Memory
from src.common.stream import Experience
from src.common.logger import Logger


class Replay(Algorithm):
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
        replay_loss_coef: float = 1.0,
        replay_batch_size: int = 32,
        memory_mode: str = "reservoir",
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
        self.memory = Memory(
            memory_size=memory_size,
            device=self.device,
            memory_seed=memory_seed,
            mode=memory_mode,
        )

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
                        mem_loss += self.criterion(pred_i, y_mem[:, i, :].unsqueeze(1))
                        x_mem = torch.concat([x_mem, pred_i], dim=1)[:, 1:, :]
                    total_loss += self.replay_loss_coef * mem_loss

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
