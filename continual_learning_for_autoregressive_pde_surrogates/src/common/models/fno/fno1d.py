import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul1d(input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # x: [B, C_in, L]
        B = x.shape[0]
        x_ft = torch.fft.rfft(x)  # [B, C_in, L//2+1]

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        # Return to physical space
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1D(nn.Module):
    """
    FNO1D with conditioning channels.

    Args:
        num_channels: number of output channels (e.g., 1 for scalar field)
        modes: number of Fourier modes
        width: spectral trunk width
        initial_step: number of context steps (T)
        cond_dim: number of conditioning features (adds N constant channels)
    """

    def __init__(
        self,
        num_channels,
        modes=16,
        width=64,
        initial_step=10,
        num_embs=0,
        emb_dim=2,
        learnable_emb=False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.initial_step = initial_step
        self.num_embs = num_embs
        self.emb_dim = emb_dim
        self.learnable_emb = learnable_emb

        # input channel count = (T*C + 1 grid + cond_dim)
        if self.num_embs > 0:
            self.fc0 = nn.Linear(
                initial_step * num_channels + 1 + self.emb_dim, self.width
            )
        else:
            self.fc0 = nn.Linear(initial_step * num_channels + 1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

        if self.num_embs > 0:
            self.embedder = nn.Embedding(
                num_embeddings=self.num_embs, embedding_dim=self.emb_dim
            )
            with torch.no_grad():
                # base = torch.linspace(0.0, 1.0, num_embeddings).unsqueeze(
                #     1
                # )  # [num_embeddings, 1]
                # noise = torch.rand(num_embeddings, embedding_dim) * 0.01  # small variation
                # self.embedder.weight[:] = (base + noise).clamp(0, 1)
                noise = torch.rand(self.num_embs, self.emb_dim)
                self.embedder.weight[:] = (noise).clamp(0, 1)
            self.embedder.requires_grad_(self.learnable_emb)

    def forward(self, x, aux_input=None, grid=None):
        # x: [B, T, L]
        # aux_input: [B, cond_dim]
        B, T, L = x.shape
        assert T == self.initial_step, f"Expected T={self.initial_step}, got T={T}"

        # Normalized spatial grid
        if grid is None:
            grid = (
                torch.linspace(0.0, 1.0, L, device=x.device, dtype=x.dtype)
                .view(1, L, 1)
                .expand(B, L, 1)
            )

        # Permute to [B, L, T]
        x = x.permute(0, 2, 1).contiguous()  # [B, L, T]

        # ---- Add conditioning as constant channels ----
        if aux_input is not None and self.num_embs > 0:
            # Expand conditioning vector across spatial dimension
            aux_input = self.embedder(aux_input.squeeze(1).to(x.device))
            cond = aux_input.unsqueeze(1).expand(B, L, self.emb_dim)  # [B, L, emb_dim]
            cond = cond.to(x.device)
            x = torch.cat((x, grid, cond), dim=-1)  # [B, L, T + 1 + emb_dim]
        else:
            x = torch.cat((x, grid), dim=-1)  # [B, L, T + 1]
        # ----------------------------------------------------

        # Lift: [B, L, channels] -> [B, L, width]
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # [B, width, L]

        # Pad spatial dimension
        x = F.pad(x, [0, self.padding])

        # 4 spectral blocks with skip connections
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Remove padding and return to [B, L, width]
        x = x[..., : -self.padding]
        x = x.permute(0, 2, 1)  # [B, L, width]

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.permute(0, 2, 1)  # [B, num_channels, L]


if __name__ == "__main__":
    # Demo
    B, T, L = 4, 10, 256  # batch, context steps, spatial points
    C_in = 1  # scalar field

    model = FNO1D(
        num_channels=C_in,
        modes=16,
        width=64,
        initial_step=T,
        num_embs=3,
        emb_dim=2,
        learnable_emb=False,
    )
    print("Total number of parameters:", sum(p.numel() for p in model.parameters()))

    x_in = torch.randn(B, T, L)
    cond = torch.LongTensor([0, 0, 1, 2]).unsqueeze(1)

    y = model(x_in, aux_input=cond)
    print("Output shape:", y.shape)  # [B, num_channels, L]
