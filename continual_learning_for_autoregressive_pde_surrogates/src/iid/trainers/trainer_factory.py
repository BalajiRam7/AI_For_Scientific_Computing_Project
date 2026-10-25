from omegaconf import DictConfig
import torch
import torch.nn as nn

from .joint_trainer import JointTrainer
from src.common.logger import Logger


def create_trainer(
    model: nn.Module,
    logger: Logger,
    device: torch.device,
    trainer_cfg: DictConfig,
):
    if trainer_cfg.trainer_name == "joint_trainer":
        return JointTrainer(
            model,
            logger,
            device,
            **trainer_cfg.trainer_parameters,
        )
    else:
        raise NotImplementedError()
