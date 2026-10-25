import yaml
import torch
import numpy as np
import random
from omegaconf import DictConfig
from omegaconf import OmegaConf


def load_yaml_file(yaml_path) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> OmegaConf:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    return cfg


def get_device(device_name: str = "cpu") -> torch.DeviceObjType:
    print("\nCompute device:", device_name)
    return torch.device(device_name)


def set_manual_seed(seed):
    print("\nSetting manual random seed to: ", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
