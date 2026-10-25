import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2dFast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer: FFT -> linear (complex) transform on low modes -> iFFT
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale
            * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    @staticmethod
    def compl_mul2d(input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, Cin, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        Hm = min(self.modes1, H)
        Wm = min(self.modes2, x_ft.size(-1))

        out_ft = torch.zeros(
            B, self.out_channels, H, x_ft.size(-1), dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :Hm, :Wm] = self.compl_mul2d(
            x_ft[:, :, :Hm, :Wm], self.weights1[:, :, :Hm, :Wm]
        )
        out_ft[:, :, -Hm:, :Wm] = self.compl_mul2d(
            x_ft[:, :, -Hm:, :Wm], self.weights2[:, :, :Hm, :Wm]
        )

        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2D(nn.Module):
    def __init__(
        self,
        num_channels,
        modes1=12,
        modes2=12,
        width=20,
        initial_step=10,
        padding=2,
        num_embs=0,
        emb_dim=2,
        learnable_emb=False,
    ):
        super().__init__()
        """
        2D Fourier Neural Operator with optional conditioning embeddings.

        Args:
            num_channels: number of output channels
            modes1, modes2: Fourier modes
            width: spectral trunk width
            initial_step: number of context time steps
            num_embs: number of conditioning embeddings (0 to disable)
            emb_dim: embedding dimension
            learnable_emb: if True, embeddings are trainable
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.initial_step = initial_step
        self.num_embs = num_embs
        self.emb_dim = emb_dim
        self.learnable_emb = learnable_emb

        # +2 for spatial grid coords, +emb_dim for conditioning if enabled
        if self.num_embs > 0:
            self.fc0 = nn.Linear(initial_step * num_channels + 2 + emb_dim, width)
        else:
            self.fc0 = nn.Linear(initial_step * num_channels + 2, width)

        self.conv0 = SpectralConv2dFast(width, width, modes1, modes2)
        self.conv1 = SpectralConv2dFast(width, width, modes1, modes2)
        self.conv2 = SpectralConv2dFast(width, width, modes1, modes2)
        self.conv3 = SpectralConv2dFast(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, kernel_size=1)
        self.w1 = nn.Conv2d(width, width, kernel_size=1)
        self.w2 = nn.Conv2d(width, width, kernel_size=1)
        self.w3 = nn.Conv2d(width, width, kernel_size=1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

        # optional conditioning embedding
        if self.num_embs > 0:
            self.embedder = nn.Embedding(num_embeddings=num_embs, embedding_dim=emb_dim)
            with torch.no_grad():
                noise = torch.rand(num_embs, emb_dim)
                self.embedder.weight[:] = noise.clamp(0, 1)
            self.embedder.requires_grad_(learnable_emb)

    @staticmethod
    def _make_grid(bsz, H, W, device):
        gx = (
            torch.linspace(0.0, 1.0, H, device=device)
            .view(1, H, 1, 1)
            .repeat(bsz, 1, W, 1)
        )
        gy = (
            torch.linspace(0.0, 1.0, W, device=device)
            .view(1, 1, W, 1)
            .repeat(bsz, H, 1, 1)
        )
        return torch.cat([gx, gy], dim=-1)

    def forward(self, x, aux_input=None):
        """
        Args:
            x: (B, T, H, W) or (B, T, C, H, W)
            aux_input: (B, 1) LongTensor indices for conditioning (optional)
        """
        if x.dim() == 4:
            B, T, H, W = x.shape
            C = 1
            x_feat = x.permute(0, 2, 3, 1).contiguous()  # (B,H,W,T)
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            x_feat = x.permute(0, 3, 4, 1, 2).contiguous().view(B, H, W, T * C)
        else:
            raise ValueError(
                f"Expected x with shape (B,T,H,W) or (B,T,C,H,W), got {tuple(x.shape)}"
            )

        # coords
        grid = self._make_grid(B, H, W, x.device)

        # optional conditioning
        if aux_input is not None and self.num_embs > 0:
            aux_input = self.embedder(aux_input.squeeze(1).to(x.device))
            cond = aux_input.view(B, 1, 1, self.emb_dim).expand(B, H, W, self.emb_dim)
            x_feat = torch.cat([x_feat, grid, cond], dim=-1)
        else:
            x_feat = torch.cat([x_feat, grid], dim=-1)

        # lift
        x_feat = self.fc0(x_feat)
        x_feat = x_feat.permute(0, 3, 1, 2).contiguous()

        # pad
        x_feat = F.pad(x_feat, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x_feat)
        x_feat = F.gelu(x1 + self.w0(x_feat))

        x1 = self.conv1(x_feat)
        x_feat = F.gelu(x1 + self.w1(x_feat))

        x1 = self.conv2(x_feat)
        x_feat = F.gelu(x1 + self.w2(x_feat))

        x1 = self.conv3(x_feat)
        x_feat = x1 + self.w3(x_feat)

        # unpad
        if self.padding > 0:
            x_feat = x_feat[..., : -self.padding, : -self.padding]

        # projection
        x_feat = x_feat.permute(0, 2, 3, 1).contiguous()
        x_feat = F.gelu(self.fc1(x_feat))
        x_feat = self.fc2(x_feat)

        out = x_feat.permute(0, 3, 1, 2).contiguous()

        if x.dim() == 5:
            out = out.unsqueeze(1)
        return out


if __name__ == "__main__":
    b, t, h, w = 2, 6, 32, 48
    c = 1
    modes1 = min(12, h)
    modes2 = min(12, w // 2 + 1)

    model = FNO2D(
        num_channels=c,
        modes1=modes1,
        modes2=modes2,
        width=32,
        initial_step=t,
        num_embs=3,
        emb_dim=2,
        learnable_emb=False,
    )

    x_in = torch.randn(b, t, h, w)
    cond = torch.LongTensor([0, 1]).unsqueeze(1)
    with torch.no_grad():
        y = model(x_in, aux_input=cond)
        print("Output shape:", y.shape)
