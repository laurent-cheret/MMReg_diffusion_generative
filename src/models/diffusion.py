"""
Simple Diffusion Model for Latent Space.

A lightweight UNet-based diffusion model that operates on VAE latents (32x32x4).
Uses DDPM (Denoising Diffusion Probabilistic Models) formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.

    Args:
        timesteps: (B,) tensor of timesteps
        embedding_dim: Dimension of embedding

    Returns:
        (B, embedding_dim) tensor of embeddings
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add timestep embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(v, attn.transpose(1, 2))
        h = h.reshape(B, C, H, W)
        h = self.proj(h)

        return x + h


class SimpleUNet(nn.Module):
    """
    Simple UNet for latent diffusion.

    Designed for 32x32x4 latent space.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Input conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        channels = [base_channels]
        ch = base_channels

        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
                channels.append(ch)

            if i < len(channel_mult) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                channels.append(ch)

        # Middle
        self.middle_block1 = ResBlock(ch, ch, time_emb_dim)
        self.middle_attn = AttentionBlock(ch)
        self.middle_block2 = ResBlock(ch, ch, time_emb_dim)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for j in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                self.decoder_blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim))
                ch = out_ch

            if i > 0:
                self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        # Output
        self.output_norm = nn.GroupNorm(8, ch)
        self.output_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy latents (B, 4, 32, 32)
            t: Timesteps (B,)

        Returns:
            Predicted noise (B, 4, 32, 32)
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Input
        h = self.input_conv(x)

        # Encoder
        skips = [h]
        block_idx = 0
        downsample_idx = 0

        num_levels = len(self.channel_mult)
        for level_idx in range(num_levels):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb)
                skips.append(h)
                block_idx += 1

            if level_idx < num_levels - 1:
                h = self.downsample[downsample_idx](h)
                skips.append(h)
                downsample_idx += 1

        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)

        # Decoder
        block_idx = 0
        upsample_idx = 0

        for level_idx in range(num_levels):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1

            if level_idx < num_levels - 1:
                h = self.upsample[upsample_idx](h)
                upsample_idx += 1

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h


class MLPDenoiser(nn.Module):
    """
    MLP-based denoiser for flat latent vectors (e.g. 256-d bottleneck).

    Used instead of UNet when diffusion operates on 1D vectors.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        time_emb_dim: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.time_emb_dim = time_emb_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual MLP blocks with time conditioning
        self.blocks = nn.ModuleList()
        self.time_projs = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.time_projs.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, hidden_dim)
            ))

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy vectors (B, D)
            t: Timesteps (B,)

        Returns:
            Predicted noise (B, D)
        """
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.input_proj(x)

        for block, time_proj in zip(self.blocks, self.time_projs):
            h = h + block(h + time_proj(t_emb))

        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_proj(h)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (from DiT paper)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # adaLN-Zero: 6 modulation parameters (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-initialize the modulation output so blocks start as identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: (B, hidden_size) conditioning vector
        mod = self.adaLN_modulation(c).unsqueeze(1)  # (B, 1, 6*D)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Self-attention with adaLN
        h = self.norm1(x) * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h)
        x = x + alpha1 * h

        # MLP with adaLN
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.mlp(h)
        x = x + alpha2 * h

        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) for latent diffusion.

    Patchifies spatial latents into a token sequence, processes with
    transformer blocks using adaLN-Zero timestep conditioning, then
    unpatchifies back to spatial output.

    DiT-S config: hidden_size=384, depth=12, num_heads=6
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        input_size: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, hidden_size)

        # Positional embedding (fixed sinusoidal or learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer: adaLN + linear projection back to patch space
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.final_linear = nn.Linear(hidden_size, patch_dim)

        # Zero-init final layers
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, patch_dim)"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, self.num_patches, -1)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch_dim) -> (B, C, H, W)"""
        B = x.shape[0]
        p = self.patch_size
        h = w = self.input_size // p
        C = self.in_channels
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        x = x.reshape(B, C, self.input_size, self.input_size)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy latents (B, C, H, W)
            t: Timesteps (B,)
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Timestep conditioning
        t_emb = get_timestep_embedding(t, self.hidden_size)
        c = self.time_embed(t_emb)  # (B, hidden_size)

        # Patchify + embed
        x = self.patchify(x)          # (B, N, patch_dim)
        x = self.patch_embed(x)       # (B, N, hidden_size)
        x = x + self.pos_embed        # add positional embedding

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer with adaLN
        mod = self.final_modulation(c).unsqueeze(1)
        gamma, beta = mod.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + gamma) + beta
        x = self.final_linear(x)      # (B, N, patch_dim)

        # Unpatchify back to spatial
        x = self.unpatchify(x)         # (B, C, H, W)
        return x


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule from Improved DDPM (Nichol & Dhariwal, 2021).
    Provides more uniform SNR distribution across timesteps than linear.
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f_t / f_t[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999).float()


class GaussianDiffusion:
    """
    DDPM Gaussian Diffusion.

    Implements the forward (noising) and reverse (denoising) processes.
    Works with both spatial (B, C, H, W) and flat (B, D) inputs.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = 'linear',
        device: str = 'cuda'
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps).to(device)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def _expand_coeffs(self, coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand coefficients to match input dimensionality (2D or 4D)."""
        shape = [coeffs.shape[0]] + [1] * (x.ndim - 1)
        return coeffs.view(*shape)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: q(x_t | x_0).

        Args:
            x_0: Clean data (B, C, H, W) or (B, D)
            t: Timesteps (B,)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy data, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._expand_coeffs(self.sqrt_alphas_cumprod[t], x_0)
        sqrt_one_minus_alpha = self._expand_coeffs(self.sqrt_one_minus_alphas_cumprod[t], x_0)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def p_losses(self, model: nn.Module, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            model: Denoising model
            x_0: Clean latents
            t: Timesteps

        Returns:
            MSE loss
        """
        x_t, noise = self.q_sample(x_0, t)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Reverse process: sample x_{t-1} from x_t.
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        predicted_noise = model(x_t, t_tensor)

        # Get coefficients
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Compute mean
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = beta / self.sqrt_one_minus_alphas_cumprod[t]
        mean = coef1 * (x_t - coef2 * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            x_t_minus_1 = mean + variance * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], progress: bool = True) -> torch.Tensor:
        """
        Generate samples by running the full reverse process.

        Args:
            model: Denoising model
            shape: Shape of samples to generate (B, C, H, W)
            progress: Show progress bar

        Returns:
            Generated latents
        """
        model.eval()
        x = torch.randn(shape, device=self.device)

        timesteps = reversed(range(self.num_timesteps))
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling", total=self.num_timesteps)

        for t in timesteps:
            x = self.p_sample(model, x, t)

        return x
