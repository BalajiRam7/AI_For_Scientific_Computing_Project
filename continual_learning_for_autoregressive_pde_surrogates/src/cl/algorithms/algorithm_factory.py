import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.common.logger import Logger
from .algorithm import Algorithm
from .naive import Naive

from .ewc import EWC
from .replay import Replay
from .gem import GEM
from .a_gem import AGEM
from .replay_ts import ReplayTS


def create_algorithm(
    model: nn.Module,
    logger: Logger,
    device: torch.DeviceObjType,
    algorithm_cfg: DictConfig,
) -> Algorithm:
    if algorithm_cfg.algorithm_name == "naive":
        return Naive(
            model=model,
            device=device,
            logger=logger,
            **algorithm_cfg.algorithm_parameters,
        )
    elif algorithm_cfg.algorithm_name == "ewc":
        return EWC(
            model=model,
            device=device,
            logger=logger,
            **algorithm_cfg.algorithm_parameters,
        )
    elif algorithm_cfg.algorithm_name == "replay":
        return Replay(
            model=model,
            device=device,
            logger=logger,
            **algorithm_cfg.algorithm_parameters,
        )
    elif algorithm_cfg.algorithm_name == "gem":
        return GEM(
            model=model,
            device=device,
            logger=logger,
            **algorithm_cfg.algorithm_parameters,
        )
    elif algorithm_cfg.algorithm_name == "a_gem":
        return AGEM(
            model=model,
            device=device,
            logger=logger,
            **algorithm_cfg.algorithm_parameters,
        )
    elif algorithm_cfg.algorithm_name == "replay_ts":
        return ReplayTS(
            model=model,
            device=device,
            logger=logger,
            **algorithm_cfg.algorithm_parameters,
        )
    else:
        raise NotImplementedError()
