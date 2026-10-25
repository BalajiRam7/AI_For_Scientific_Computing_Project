import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .algorithm import Algorithm
from src.common.stream import Experience
from src.common.logger import Logger


class Naive(Algorithm):
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

    def train_exp(self, experience: Experience):
        self.model.train()
        train_loader = DataLoader(
            experience.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        for epoch in range(self.epochs):
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in train_progress:
                x, y = batch["input"].to(self.device), batch["output"].to(self.device)
                n_rollout_steps = y.shape[1]
                self.optimizer.zero_grad()
                loss = 0.0
                for i in range(n_rollout_steps):
                    pred_i = self.model(x, aux_input=batch["aux_input"])
                    loss += self.criterion(pred_i, y[:, i, :].unsqueeze(1))
                    x = torch.concat([x, pred_i], dim=1)[:, 1:, :]
                loss.backward()
                self.optimizer.step()

                # Logging
                self.log_metric(phase="train", key="step_loss", value=loss.item())
                self.global_step += 1

        self.log_metric(phase="train", key="exp_idx", value=experience.exp_idx)
