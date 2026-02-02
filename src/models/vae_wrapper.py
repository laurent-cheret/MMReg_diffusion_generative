"""
VAE Wrapper for MM-Reg training.

Wraps diffusers AutoencoderKL to add MM-Reg loss computation.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Optional, Dict, Any


class MMRegVAE(nn.Module):
    """
    VAE wrapper that adds MM-Reg functionality to AutoencoderKL.

    Provides methods for:
    - Encoding images to latents
    - Decoding latents to images
    - Computing flattened latents for MM-Reg
    """

    def __init__(
        self,
        pretrained_path: str = "stabilityai/sd-vae-ft-mse",
        use_gradient_checkpointing: bool = True
    ):
        """
        Args:
            pretrained_path: Path or HF model ID for pretrained VAE
            use_gradient_checkpointing: Enable gradient checkpointing for memory
        """
        super().__init__()

        # Load pretrained VAE
        self.vae = AutoencoderKL.from_pretrained(pretrained_path)

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            try:
                # New diffusers API
                self.vae.enable_gradient_checkpointing()
            except AttributeError:
                # Fallback for older versions
                pass

        # Store config
        self.latent_channels = self.vae.config.latent_channels  # Usually 4
        self.scaling_factor = self.vae.config.scaling_factor  # Usually 0.18215

    def encode(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode images to latent space.

        Args:
            x: Images in range [-1, 1], shape (B, 3, H, W)
            sample: If True, sample from posterior; else return mean

        Returns:
            Dictionary with:
            - 'latent': Latent tensor (B, C, H/8, W/8)
            - 'latent_flat': Flattened latent for MM-Reg (B, C*H/8*W/8)
            - 'posterior': DiagonalGaussianDistribution for KL
        """
        # Encode to posterior
        posterior = self.vae.encode(x).latent_dist

        # Sample or use mean
        if sample:
            latent = posterior.sample()
        else:
            latent = posterior.mean

        # Scale latents (SD convention)
        latent = latent * self.scaling_factor

        # Flatten for MM-Reg (use reshape for non-contiguous tensors)
        latent_flat = latent.reshape(latent.shape[0], -1)

        return {
            'latent': latent,
            'latent_flat': latent_flat,
            'posterior': posterior
        }

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latent: Latent tensor (B, C, H/8, W/8), scaled

        Returns:
            Reconstructed images in range [-1, 1]
        """
        # Unscale latents
        latent = latent / self.scaling_factor

        # Decode
        decoded = self.vae.decode(latent).sample

        return decoded

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> decode.

        Args:
            x: Images in range [-1, 1], shape (B, 3, H, W)
            sample: If True, sample from posterior

        Returns:
            Dictionary with all outputs
        """
        # Encode
        enc_out = self.encode(x, sample=sample)

        # Decode
        x_recon = self.decode(enc_out['latent'])

        return {
            'x_recon': x_recon,
            'latent': enc_out['latent'],
            'latent_flat': enc_out['latent_flat'],
            'posterior': enc_out['posterior']
        }

    def get_latent_shape(self, image_size: int = 256) -> tuple:
        """Get the shape of latent for given image size."""
        latent_size = image_size // 8  # VAE downsamples by 8
        return (self.latent_channels, latent_size, latent_size)

    def freeze_decoder(self):
        """Freeze decoder parameters (for encoder-only finetuning)."""
        for param in self.vae.decoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.vae.parameters():
            param.requires_grad = True


def load_vae(
    pretrained_path: str = "stabilityai/sd-vae-ft-mse",
    device: str = "cuda",
    use_gradient_checkpointing: bool = True
) -> MMRegVAE:
    """
    Load VAE model.

    Args:
        pretrained_path: HuggingFace model ID or local path
        device: Device to load model on
        use_gradient_checkpointing: Enable memory optimization

    Returns:
        MMRegVAE model
    """
    model = MMRegVAE(
        pretrained_path=pretrained_path,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    model = model.to(device)
    return model
