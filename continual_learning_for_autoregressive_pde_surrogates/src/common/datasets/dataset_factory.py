from typing import Optional

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from .advdiff1d import AdvDiff1D
from .advdiff2d import AdvDiff2D
from .burgers1d import Burgers1D
from .burgers2d import Burgers2D
from .navierstokes2d import NavierStokes2D


def _resolve_dataset_params(
    dataset_cfg: DictConfig, rollout_steps: Optional[int] = None
) -> dict:
    params = OmegaConf.to_container(dataset_cfg.dataset_params, resolve=True)
    params.pop("rollout_steps_test", None)
    if rollout_steps is not None:
        params["rollout_steps"] = rollout_steps
    return params


def create_dataset(
    dataset_cfg: DictConfig,
    file_list: list = None,
    rollout_steps: Optional[int] = None,
) -> Dataset:
    params = _resolve_dataset_params(dataset_cfg, rollout_steps=rollout_steps)
    if dataset_cfg.dataset_name == "advdiff1d":
        return AdvDiff1D(**params, file_list=file_list)
    elif dataset_cfg.dataset_name == "advdiff2d":
        return AdvDiff2D(**params, file_list=file_list)
    elif dataset_cfg.dataset_name == "burgers1d":
        return Burgers1D(**params, file_list=file_list)
    elif dataset_cfg.dataset_name == "burgers2d":
        return Burgers2D(**params, file_list=file_list)
    elif dataset_cfg.dataset_name == "navierstokes2d":
        return NavierStokes2D(**params, file_list=file_list)
    else:
        raise NotImplementedError()
