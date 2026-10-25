from collections import OrderedDict
import torch
from torch import nn


class UNet1DCore(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()

        features = init_features
        self.encoder1 = UNet1DCore._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1DCore._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1DCore._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1DCore._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1DCore._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1DCore._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1DCore._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1DCore._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1DCore._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # Use ReLU activations in blocks; apply Kaiming init suited for ReLU
        self.apply(self._init_weights_kaiming_relu)
        # Zero-init final projection to start near-zero predictions for stability
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # Pad if needed
        remainder = x.shape[-1] % 16
        pad_len = (16 - remainder) if remainder != 0 else 0
        if pad_len:
            x = nn.functional.pad(x, (0, pad_len))

        # Regular feed-forward
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)

        # Crop back to the original size
        if pad_len:
            out = out[..., :-pad_len]

        return out

    @staticmethod
    def _block(in_channels, features, name):
        # No normalization, to match UNet2D and the FNO models. BatchNorm here
        # destroys the absolute-scale information needed for this autoregressive
        # regression task and severely degrades convergence.
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def _init_weights_kaiming_relu(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class UNet1D(nn.Module):
    """
    FNO-aligned wrapper around UNet1DCore.

    - Input x: [B, T, L]
    - Optional aux_input: [B, 1] Long indices for conditioning
    - Adds 1 grid channel and optional embedding channels
    - Output: [B, num_channels, L]
    """

    def __init__(
        self,
        num_channels: int = 1,
        initial_step: int = 10,
        init_features: int = 32,
        num_embs: int = 0,
        emb_dim: int = 2,
        learnable_emb: bool = False,
        # Back-compat: accept legacy args but ignore (in_channels/out_channels)
        in_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.initial_step = initial_step
        self.num_embs = num_embs
        self.emb_dim = emb_dim
        self.learnable_emb = learnable_emb

        if self.num_embs > 0:
            self.embedder = nn.Embedding(num_embeddings=self.num_embs, embedding_dim=self.emb_dim)
            with torch.no_grad():
                noise = torch.rand(self.num_embs, self.emb_dim)
                self.embedder.weight[:] = noise.clamp(0, 1)
            self.embedder.requires_grad_(self.learnable_emb)
        else:
            self.embedder = None

        core_in_ch = initial_step * num_channels + 1 + (self.emb_dim if self.num_embs > 0 else 0)
        self.core = UNet1DCore(in_channels=core_in_ch, out_channels=num_channels, init_features=init_features)

    def forward(self, x: torch.Tensor, aux_input: torch.Tensor | None = None, grid: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, L]
        B, T, L = x.shape
        assert T == self.initial_step, f"Expected T={self.initial_step}, got T={T}"

        if grid is None:
            grid = (
                torch.linspace(0.0, 1.0, L, device=x.device, dtype=x.dtype)
                .view(1, L, 1)
                .expand(B, L, 1)
            )

        feat = x.permute(0, 2, 1).contiguous()  # [B, L, T]

        if aux_input is not None and self.num_embs > 0 and self.embedder is not None:
            emb = self.embedder(aux_input.squeeze(1).to(x.device))  # [B, emb_dim]
            cond = emb.unsqueeze(1).expand(B, L, self.emb_dim)
            feat = torch.cat([feat, grid, cond], dim=-1)
        else:
            feat = torch.cat([feat, grid], dim=-1)

        feat = feat.permute(0, 2, 1).contiguous()  # [B, C_in, L]
        out = self.core(feat)  # [B, num_channels, L]
        return out


if __name__ == "__main__":
    # Smoke test aligned with FNO1D/GNO1D wrapper API
    B, T, L = 4, 10, 256
    C_in = 1

    model = UNet1D(
        num_channels=C_in,
        initial_step=T,
        init_features=16,
        num_embs=3,
        emb_dim=2,
        learnable_emb=False,
    )

    print("Total number of parameters:", sum(p.numel() for p in model.parameters()))

    x_in = torch.randn(B, T, L)
    cond = torch.LongTensor([0, 0, 1, 2]).unsqueeze(1)
    with torch.no_grad():
        y = model(x_in, aux_input=cond)
    print("Output shape:", y.shape)  # [B, C_in, L]
