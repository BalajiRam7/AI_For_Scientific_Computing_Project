import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2DCore(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
        num_down: int = 4,
    ):
        super().__init__()
        self.num_down = int(num_down)

        # Encoder
        enc_blocks = []
        pools = []
        ch_in = in_channels
        ch = base_channels
        for _ in range(self.num_down):
            enc_blocks.append(ConvBlock(ch_in, ch))
            pools.append(nn.MaxPool2d(2))
            ch_in = ch
            ch = ch * 2
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.pools = nn.ModuleList(pools)

        # Bottleneck
        self.bottleneck = ConvBlock(ch_in, ch_in)

        # Decoder
        dec_blocks = []
        ups = []
        dec_ch_in = ch_in
        for i in range(self.num_down - 1, -1, -1):
            skip_ch = base_channels * (2**i)
            ups.append(nn.ConvTranspose2d(dec_ch_in, skip_ch, kernel_size=2, stride=2))
            dec_blocks.append(ConvBlock(skip_ch + skip_ch, skip_ch))
            dec_ch_in = skip_ch
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.out_conv = nn.Conv2d(dec_ch_in, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H_orig, W_orig = x.shape

        # Compute padding to make sizes divisible by 2^num_down
        factor = 2**self.num_down
        pad_h_total = (factor - (H_orig % factor)) % factor
        pad_w_total = (factor - (W_orig % factor)) % factor
        if pad_h_total > 0 or pad_w_total > 0:
            x = F.pad(x, (0, pad_w_total, 0, pad_h_total))  # (left,right,top,bottom)

        def center_crop(
            feat: torch.Tensor, target_h: int, target_w: int
        ) -> torch.Tensor:
            _, _, h, w = feat.shape
            if h == target_h and w == target_w:
                return feat
            dh = max(0, h - target_h)
            dw = max(0, w - target_w)
            top = dh // 2
            left = dw // 2
            return feat[:, :, top : top + target_h, left : left + target_w]

        # Encoder forward with feature caching for skips
        skips = []
        h = x
        for i in range(self.num_down):
            h = self.enc_blocks[i](h)
            skips.append(h)
            h = self.pools[i](h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder with skip connections
        for i in range(self.num_down - 1, -1, -1):
            h = self.ups[self.num_down - 1 - i](h)
            skip = skips[i]
            skip_c = center_crop(skip, h.shape[2], h.shape[3])
            h = torch.cat([h, skip_c], dim=1)
            h = self.dec_blocks[self.num_down - 1 - i](h)

        out = self.out_conv(h)  # [B,out_ch,H_pad,W_pad]
        # Crop back to original spatial size if padded
        out = out[:, :, :H_orig, :W_orig]
        return out


class UNet2D(nn.Module):
    """
    FNO-aligned wrapper around UNet2DCore.

    - Input x: [B, T, H, W] or [B, T, C, H, W]
    - Optional aux_input: [B, 1] Long indices for conditioning
    - Adds 2 grid channels and optional embedding channels
    - Output: [B, num_channels, H, W] for 4D input,
              [B, 1, num_channels, H, W] for 5D input (matches FNO2D)
    """

    def __init__(
        self,
        num_channels: int = 1,
        initial_step: int = 4,
        base_channels: int = 32,
        num_down: int = 4,
        num_embs: int = 0,
        emb_dim: int = 2,
        learnable_emb: bool = False,
        # Back-compat
        in_channels: int | None = None,
        out_dim: list[int] | None = None,
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

        # core input channels = T*num_channels + 2 (grid) + emb_dim (if enabled)
        core_in_ch = (
            initial_step * num_channels
            + 2
            + (self.emb_dim if self.num_embs > 0 else 0)
        )
        self.core = UNet2DCore(
            in_channels=core_in_ch,
            out_channels=num_channels,
            base_channels=base_channels,
            num_down=num_down,
        )

    @staticmethod
    def _make_grid(bsz: int, H: int, W: int, device) -> torch.Tensor:
        gx = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1, 1).repeat(bsz, 1, W, 1)
        gy = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, W, 1).repeat(bsz, H, 1, 1)
        return torch.cat([gx, gy], dim=-1)  # [B,H,W,2]

    def forward(self, x: torch.Tensor, aux_input: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B,T,H,W) or (B,T,C,H,W)
        if x.dim() == 4:
            B, T, H, W = x.shape
            C_in = 1
            x_feat = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,T]
        elif x.dim() == 5:
            B, T, C_in, H, W = x.shape
            x_feat = x.permute(0, 3, 4, 1, 2).contiguous().view(B, H, W, T * C_in)
        else:
            raise ValueError(f"Expected x with shape (B,T,H,W) or (B,T,C,H,W), got {tuple(x.shape)}")

        assert T == self.initial_step, f"Expected T={self.initial_step}, got T={T}"

        grid = self._make_grid(B, H, W, x.device)

        if aux_input is not None and self.num_embs > 0 and self.embedder is not None:
            emb = self.embedder(aux_input.squeeze(1).to(x.device))  # [B, emb_dim]
            cond = emb.view(B, 1, 1, self.emb_dim).expand(B, H, W, self.emb_dim)
            x_feat = torch.cat([x_feat, grid, cond], dim=-1)
        else:
            x_feat = torch.cat([x_feat, grid], dim=-1)

        x_feat = x_feat.permute(0, 3, 1, 2).contiguous()  # [B,C_in,H,W]
        out = self.core(x_feat)  # [B,num_channels,H,W]

        if x.dim() == 5:
            out = out.unsqueeze(1)  # [B,1,num_channels,H,W] to match (B,T,C,H,W) convention
        return out


if __name__ == "__main__":
    # Smoke test aligned with FNO2D-style wrapper API
    B, T, H, W = 2, 4, 64, 48
    C_in = 1

    model = UNet2D(
        num_channels=C_in,
        initial_step=T,
        base_channels=16,
        num_down=3,
        num_embs=3,
        emb_dim=2,
        learnable_emb=False,
    )

    print("Total number of parameters:", sum(p.numel() for p in model.parameters()))

    x_in = torch.randn(B, T, H, W)
    cond = torch.LongTensor([0, 1]).unsqueeze(1)
    with torch.no_grad():
        y = model(x_in, aux_input=cond)
    print("Output shape:", y.shape)  # [B, C_in, H, W]
