from omegaconf import DictConfig
import torch.nn as nn
import torch

from .unet.unet1d import UNet1D
from .unet.unet2d import UNet2D
from .fno.fno1d import FNO1D
from .fno.fno2d import FNO2D


def create_model(
    device: torch.DeviceObjType,
    model_cfg: DictConfig,
    checkpoint_path: str = None,
) -> nn.Module:
    if model_cfg.model_name == "unet1d":
        model = UNet1D(**model_cfg.model_parameters).to(device)
    elif model_cfg.model_name == "unet2d":
        model = UNet2D(**model_cfg.model_parameters).to(device)
    elif model_cfg.model_name == "fno1d":
        model = FNO1D(**model_cfg.model_parameters).to(device)
    elif model_cfg.model_name == "fno2d":
        model = FNO2D(**model_cfg.model_parameters).to(device)
    else:
        raise NotImplementedError()

    # Load checkpoint if given
    if checkpoint_path:
        print(f"\nLoading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        # model.load_state_dict(state_dict, strict=False)

        for name, param in model.named_parameters():
            if name in state_dict:
                ckpt_param = state_dict[name]
                if ckpt_param.shape == param.shape:
                    param.data.copy_(ckpt_param)
                else:
                    print(f"Skipping parameter because of size mismatch: {name}")
            else:
                print(f"Param {name} not found in the checkpoint.")
    return model
