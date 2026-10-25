import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpectralLoss(nn.Module):
    """
    Frequency-space (spectral) loss.

    Args:
        magnitude_only (bool): if True, compares |FFT| instead of full complex FFT.
        freq_weight (torch.Tensor, optional): optional frequency weighting tensor, same shape as FFT output.
    """

    def __init__(self, magnitude_only=False, freq_weight=None):
        super().__init__()
        self.magnitude_only = magnitude_only
        self.freq_weight = freq_weight

    def forward(self, pred, target, step=None):
        pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
        target_fft = torch.fft.fftn(target, dim=(-2, -1))

        if self.magnitude_only:
            diff = torch.abs(pred_fft) - torch.abs(target_fft)
        else:
            diff = pred_fft - target_fft

        spectral_loss = torch.abs(diff) ** 2

        if self.freq_weight is not None:
            spectral_loss = spectral_loss * self.freq_weight.to(pred.device)

        spectral_loss = spectral_loss.mean()

        return spectral_loss


class MSEandSpectraLoss(nn.Module):
    """
    Combines pixel-space MSE loss with frequency-space (spectral) loss.

    Args:
        spectral_weight (float or callable): weight λ_s for spectral loss term.
            Can be a scalar (e.g. 0.1) or a callable returning a scalar (e.g. lambda step: 0.1 * (1 - step/10000)).
        magnitude_only (bool): if True, compares |FFT| instead of full complex FFT.
        freq_weight (torch.Tensor, optional): optional frequency weighting tensor, same shape as FFT output.
    """

    def __init__(self, spectral_weight=0.1, magnitude_only=False, freq_weight=None):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.magnitude_only = magnitude_only
        self.freq_weight = freq_weight

        self.log_dict = {"mse_loss": 0.0, "spectral_loss": 0.0}

    def forward(self, pred, target, step=None):
        # ---- 1. Spatial MSE loss ----
        mse_loss = F.mse_loss(pred, target)

        # ---- 2. Spectral loss ----
        pred_fft = torch.fft.fftn(pred, dim=(-2, -1))
        target_fft = torch.fft.fftn(target, dim=(-2, -1))

        if self.magnitude_only:
            diff = torch.abs(pred_fft) - torch.abs(target_fft)
        else:
            diff = pred_fft - target_fft

        spectral_loss = torch.abs(diff) ** 2

        if self.freq_weight is not None:
            spectral_loss = spectral_loss * self.freq_weight.to(pred.device)

        spectral_loss = spectral_loss.mean()

        # ---- 3. Combine with weighting ----
        lambda_s = self.spectral_weight

        total_loss = mse_loss + lambda_s * spectral_loss

        # Set log items
        self.log_dict["mse_loss"] = mse_loss.item()
        self.log_dict["spectral_loss"] = spectral_loss.item()

        return total_loss


class MSEandPerceptualLoss(nn.Module):
    def __init__(self, layer_name="relu2_2", perceptual_weight=0.1):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.vgg_layers = vgg
        self.layer_name = layer_name
        self.perceptual_weight = perceptual_weight
        self.layer_map = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu4_1": 20,
            "relu4_2": 22,
        }
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.log_dict = {"mse_loss": 0.0, "perceptual_loss": 0.0}

    def forward(self, pred, target):
        # Normalize input to match VGG expectations
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        layer_idx = self.layer_map[self.layer_name]

        # Extract up to the desired layer
        feat_pred = self.vgg_layers[: layer_idx + 1](pred_norm)
        feat_target = self.vgg_layers[: layer_idx + 1](target_norm)

        mse_loss = F.mse_loss(pred, target)
        perceptual_loss = F.mse_loss(feat_pred, feat_target)

        total_loss = mse_loss + self.perceptual_weight * perceptual_loss

        self.log_dict["mse_loss"] = mse_loss.item()
        self.log_dict["perceptual_loss"] = perceptual_loss.item()

        return total_loss
