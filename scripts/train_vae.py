"""
Training script for MM-Reg VAE.

Usage:
    python scripts/train_vae.py --config configs/vae/mmreg_dinov2.yaml
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_wrapper import load_vae
from src.models.reference import get_reference_model
from src.models.losses import VAELoss
from src.data.dataset import get_dataset_and_loader
from src.trainer import MMRegTrainer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train MM-Reg VAE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("MM-Reg VAE Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Lambda MM: {config['loss']['lambda_mm']}")
    print("=" * 60)

    # Load VAE
    print("\nLoading VAE...")
    vae = load_vae(
        pretrained_path=config['model']['pretrained_path'],
        device=device,
        use_gradient_checkpointing=config['model']['use_gradient_checkpointing']
    )

    # Load reference model
    print("Loading reference model...")
    reference_model = get_reference_model(
        ref_type=config['reference']['type'],
        device=device,
        model_name=config['reference'].get('model_name', 'dinov2_vitb14')
    )

    # Create loss function
    loss_fn = VAELoss(
        lambda_mm=config['loss']['lambda_mm'],
        beta=config['loss']['beta'],
        mm_variant=config['loss'].get('mm_variant', 'correlation'),
        reconstruction_type=config['loss']['reconstruction_type']
    )

    # Load data
    print("Loading data...")
    train_dataset, train_loader = get_dataset_and_loader(
        dataset_name=config['data']['dataset'],
        root=config['data']['root'],
        split='train',
        image_size=config['data']['image_size'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    val_dataset, val_loader = get_dataset_and_loader(
        dataset_name=config['data']['dataset'],
        root=config['data']['root'],
        split='val',
        image_size=config['data']['image_size'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler (linear warmup + cosine decay)
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = config['training']['warmup_steps']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create trainer
    trainer = MMRegTrainer(
        vae=vae,
        reference_model=reference_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_amp=config['training']['use_amp'],
        log_interval=config['training']['log_interval'],
        save_dir=config['output']['save_dir'],
        scheduler=scheduler
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        num_epochs=config['training']['epochs'],
        save_every=config['training']['save_every']
    )


if __name__ == '__main__':
    main()
