import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from omegaconf import DictConfig
import numpy as np

from .trainer import Trainer
from src.common.logger import Logger
from src.common.stream import Stream
from src.common.visualization.sample_grid import generate_and_plot_sample_grid
from src.common.visualization.visualization import plot_bar_using_dict
from src.common.loss_functions import MSEandSpectraLoss, MSEandPerceptualLoss


class JointTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        logger: Logger,
        device: torch.device,
        num_workers: int = 0,
        batch_size: int = 64,
        epochs: int = 50,
        optimizer: str = "adam",
        optimizer_params: DictConfig = {},
        criterion: str = "mse",
        criterion_params: DictConfig = {},
        scheduler_params: DictConfig = None,
    ):
        super().__init__(device, logger, num_workers)
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

        # Optimizer
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        else:
            raise NotImplementedError()

        if criterion == "mse":
            self.criterion = nn.MSELoss(**criterion_params)
        elif criterion == "mse+spectral":
            self.criterion = MSEandSpectraLoss(**criterion_params)
        elif criterion == "mse+perceptual":
            self.criterion = MSEandPerceptualLoss(**criterion_params)
            self.criterion.vgg_layers.to(self.device)
        else:
            raise NotImplementedError()
        self.scheduler = None
        if scheduler_params:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, **scheduler_params
            )

    def train(self, stream: Stream):
        self.model.train()
        self.model.to(self.device)

        combined_train_ds = torch.utils.data.ConcatDataset(
            [exp.train_dataset for exp in stream]
        )
        self.train_loader = DataLoader(
            combined_train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        for epoch in range(self.epochs):
            epoch_train_loss = 0.0
            train_progress_bar = tqdm.tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"
            )
            for batch in train_progress_bar:
                x, y = batch["input"], batch["output"]
                x, y = x.to(self.device), y.to(self.device)
                n_rollout_steps = y.shape[1]

                self.optimizer.zero_grad()
                loss = 0.0
                for i in range(n_rollout_steps):
                    pred_i = self.model(x, aux_input=batch["aux_input"])
                    loss += self.criterion(pred_i, y[:, i, :].unsqueeze(1))
                    x = torch.concat([x, pred_i], dim=1)[:, 1:, :]
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                epoch_train_loss += loss.item()
                self.log_metric(phase="train", key="step_loss", value=loss.item())
                self.global_step += 1

            epoch_train_loss /= len(self.train_loader)
            self.log_metric(phase="train", key="epoch_loss", value=epoch_train_loss)
            self.log_metric(phase="train", key="epoch", value=epoch)
            if stream[0].val_dataset:
                self.evaluate(stream, mode="val")

    @torch.no_grad()
    def eval(self, stream, mode="test"):
        stream_eval_logs = {"labels": [], "values": []}
        rel_l2_error_list = []
        for exp in stream:
            loader = DataLoader(
                exp.test_dataset if mode == "test" else exp.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            self.model.eval()
            total_loss = 0.0
            total_count = 0

            # Relative L2 Error
            rel_l2_num = 0.0
            rel_l2_den = 0.0

            for batch in loader:
                x, y = batch["input"], batch["output"]
                x, y = x.to(self.device), y.to(self.device)
                n_rollout_steps = y.shape[1]

                loss = 0.0
                for i in range(n_rollout_steps):
                    pred_i = self.model(x, aux_input=batch["aux_input"])
                    loss += self.criterion(pred_i, y[:, i, :].unsqueeze(1))
                    x = torch.concat([x, pred_i], dim=1)[:, 1:, :]

                    # Relative L2
                    rel_l2_num += torch.sum(
                        (pred_i - y[:, i, :].unsqueeze(1)) ** 2
                    ).item()
                    rel_l2_den += torch.sum(y[:, i, :] ** 2).item()

                bsize = x.shape[0]
                total_loss += loss.item() * bsize
                total_count += bsize

            avg_loss = total_loss / total_count
            rel_l2_error = (rel_l2_num / rel_l2_den) ** 0.5

            self.model.train()

            stream_eval_logs["labels"].append(f"e{exp.exp_idx}")
            stream_eval_logs["values"].append(avg_loss)
            rel_l2_error_list.append(rel_l2_error)

            self.log_metric(
                phase="eval", key=f"{mode}_loss_e{exp.exp_idx}", value=avg_loss
            )
            self.log_metric(
                phase="eval",
                key=f"{mode}_loss_rell2_e{exp.exp_idx}",
                value=rel_l2_error,
            )

        # Average eval loss on the entire stream
        avg_stream_eval_loss = np.mean(stream_eval_logs["values"])
        self.log_metric(
            phase="eval", key=f"{mode}_loss_stream", value=avg_stream_eval_loss
        )
        self.log_metric(
            phase="eval",
            key=f"{mode}_loss_rell2_stream",
            value=np.mean(rel_l2_error_list),
        )

        return stream_eval_logs

    def finalize(self, stream, stream_eval_logs):
        bar_fig = plot_bar_using_dict(stream_eval_logs)
        self.log_figure(phase="eval", fig=bar_fig, name="final_stream_eval")

        grid_fig = generate_and_plot_sample_grid(self.model, stream)
        self.log_figure(phase="eval", fig=grid_fig, name="sample_grid")

        checkpoint_path = os.path.join(
            self.logger.checkpoints_dir, "model_weights_final.pt"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.finalize_run()
