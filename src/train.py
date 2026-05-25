from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import diff_augment
from data import YugiohCardDataset
from model import Discriminator, Generator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCGAN training for Yu Gi Oh card art.")
    p.add_argument("--data-dir", type=str, default="data/raw/yugioh_cards")
    p.add_argument(
        "--manifest", type=str, default="data/raw/yugioh_cards/manifest.csv"
    )
    p.add_argument("--output-dir", type=str, default="runs/dcgan")
    p.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--d-steps", type=int, default=1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--sample-grid", type=int, default=64)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--augment",
        type=str,
        default="color,translation,cutout",
        help=(
            "DiffAugment policy applied to both real and fake batches inside "
            "the discriminator loss. Pass an empty string to disable."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from.",
    )
    return p.parse_args()


def hinge_d_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """SAGAN style hinge loss for the discriminator."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def hinge_g_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """SAGAN style hinge loss for the generator."""
    return -fake_scores.mean()


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Exponential moving average of generator weights for sample quality."""
    for ep, p in zip(ema_model.parameters(), model.parameters()):
        ep.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for eb, b in zip(ema_model.buffers(), model.buffers()):
        eb.copy_(b)


def make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """Build a GradScaler compatible across recent PyTorch versions."""
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataset = YugiohCardDataset(
        images_dir=args.data_dir,
        manifest_csv=args.manifest,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    G = Generator(
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        base_channels=args.base_channels,
    ).to(device)
    D = Discriminator(
        image_size=args.image_size,
        base_channels=args.base_channels,
    ).to(device)

    # EMA generator gives smoother, higher quality samples.
    G_ema = deepcopy(G).eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(
        G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2)
    )
    opt_d = torch.optim.Adam(
        D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2)
    )

    scaler_g = make_grad_scaler(device, enabled=use_amp)
    scaler_d = make_grad_scaler(device, enabled=use_amp)

    fixed_noise = torch.randn(args.sample_grid, args.latent_dim, 1, 1, device=device)

    # Optionally restore from a previous checkpoint.
    start_epoch = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Resuming from {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        G_ema.load_state_dict(ckpt["G_ema"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    print(
        f"device={device} samples={len(dataset)} batches_per_epoch={len(loader)} "
        f"image_size={args.image_size} amp={use_amp} start_epoch={start_epoch}",
        flush=True,
    )

    step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        sum_d = sum_g = 0.0
        seen = 0

        for real in loader:
            real = real.to(device, non_blocking=True)
            bs = real.size(0)

            # Discriminator update with hinge loss.
            for _ in range(args.d_steps):
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    z = torch.randn(bs, args.latent_dim, 1, 1, device=device)
                    with torch.no_grad():
                        fake = G(z)
                    real_aug = diff_augment(real, policy=args.augment)
                    fake_aug = diff_augment(fake, policy=args.augment)
                    d_loss = hinge_d_loss(D(real_aug), D(fake_aug))
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

            # Generator update with hinge loss.
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                z = torch.randn(bs, args.latent_dim, 1, 1, device=device)
                fake = G(z)
                fake_aug = diff_augment(fake, policy=args.augment)
                g_loss = hinge_g_loss(D(fake_aug))
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            update_ema(G_ema, G, args.ema_decay)

            sum_d += d_loss.item() * bs
            sum_g += g_loss.item() * bs
            seen += bs
            step += 1

        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch:>4d}/{args.epochs} "
            f"d_loss={sum_d / max(seen, 1):.4f} "
            f"g_loss={sum_g / max(seen, 1):.4f} "
            f"({elapsed:.1f}s, step={step})",
            flush=True,
        )

        if args.sample_every and epoch % args.sample_every == 0:
            G_ema.eval()
            with torch.no_grad():
                samples = G_ema(fixed_noise).clamp(-1, 1)
            save_image(
                (samples + 1) / 2,
                samples_dir / f"epoch_{epoch:04d}.png",
                nrow=int(args.sample_grid**0.5),
            )

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "G_ema": G_ema.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "args": vars(args),
                },
                checkpoints_dir / f"ckpt_{epoch:04d}.pt",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
