from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

from src.common.logger import Logger


class Trainer(ABC):
    def __init__(
        self,
        device: torch.device,
        logger: Logger,
        num_workers: int = 0,
    ):
        self.num_workers = num_workers
        self.device = device
        self.logger = logger
        self.global_step = 0

    @abstractmethod
    def train(self, train_ds: Dataset):
        pass

    @abstractmethod
    def eval(self, test_ds: Dataset):
        pass

    def log_metric(self, phase="train", key="step_loss", value=None):
        self.logger.log_metric(
            metric_value=value,
            metric_name=key,
            metric_context=phase,
            step=self.global_step,
        )

    def log_figure(self, phase="train", fig=None, name="sample"):
        self.logger.log_figure(
            fig,
            fig_name=name,
            fig_context=phase,
            step=self.global_step,
        )
