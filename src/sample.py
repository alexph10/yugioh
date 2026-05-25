from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Generator
from render import render_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate images from a trained DCGAN checkpoint."
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint.")
    p.add_argument("--output-dir", type=str, default="samples")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--use-live",
        action="store_true",
        help="Sample from the live generator instead of the EMA generator.",
    )
    p.add_argument(
        "--cards",
        action="store_true",
        help="Also render each sample as a styled card mockup.",
    )
    p.add_argument("--grid-cols", type=int, default=8)
    p.add_argument(
        "--truncation",
        type=float,
        default=None,
        help=(
            "Optional truncation factor in (0, 1]. Latents are resampled until "
            "their per element absolute value is below this threshold, trading "
            "diversity for fidelity."
        ),
    )
    p.add_argument(
        "--interpolate",
        action="store_true",
        help=(
            "Render a spherical interpolation (slerp) between two latent "
            "endpoints instead of sampling independent latents."
        ),
    )
    p.add_argument(
        "--interpolate-steps",
        type=int,
        default=16,
        help="Number of interpolation frames between the two endpoints.",
    )
    p.add_argument(
        "--interpolate-pairs",
        type=int,
        default=4,
        help="How many distinct endpoint pairs to interpolate.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def sample_truncated(
    n: int, latent_dim: int, device: torch.device, truncation: float | None
) -> torch.Tensor:
    """Sample z ~ N(0, I) and, if a truncation threshold is given, resample any
    element whose absolute value exceeds it. Lower values trade diversity for
    fidelity, mirroring the BigGAN truncation trick.
    """
    z = torch.randn(n, latent_dim, 1, 1, device=device)
    if truncation is None:
        return z
    if not (0.0 < truncation <= 5.0):
        raise ValueError("truncation must be in (0, 5].")
    # Rejection resample until every element is within [neg t, t]. With a
    # reasonable threshold this converges in a handful of passes.
    for _ in range(16):
        mask = z.abs() > truncation
        if not bool(mask.any()):
            break
        z = torch.where(mask, torch.randn_like(z), z)
    return z.clamp(-truncation, truncation)


def slerp(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors.

    Operates on the flattened latent so direction is preserved on the unit
    sphere; ``t`` is a 1D tensor of weights in [zero, one].
    """
    flat0 = z0.flatten(1)
    flat1 = z1.flatten(1)
    n0 = flat0 / flat0.norm(dim=1, keepdim=True).clamp_min(1e-8)
    n1 = flat1 / flat1.norm(dim=1, keepdim=True).clamp_min(1e-8)
    dot = (n0 * n1).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-8)

    t = t.to(flat0).view(-1, 1)  # (steps, 1)
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    out_flat = a * flat0 + b * flat1
    return out_flat.view(-1, *z0.shape[1:])


def build_interpolation_latents(
    pairs: int, steps: int, latent_dim: int, device: torch.device
) -> torch.Tensor:
    """Return a stack of ``pairs * steps`` latents that slerp between random
    endpoint pairs. Rows of the eventual sample grid are full interpolations.
    """
    z_pairs = torch.randn(pairs, 2, latent_dim, 1, 1, device=device)
    t = torch.linspace(0.0, 1.0, steps, device=device)
    rows: list[torch.Tensor] = []
    for i in range(pairs):
        z0 = z_pairs[i, 0:1]
        z1 = z_pairs[i, 1:2]
        row = slerp(z0.expand(steps, -1, -1, -1), z1.expand(steps, -1, -1, -1), t)
        rows.append(row)
    return torch.cat(rows, dim=0)


def load_generator(checkpoint_path: Path, device: torch.device, use_live: bool) -> tuple[Generator, dict]:
    """Reconstruct the generator using the architecture args stored in the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args") or {}

    latent_dim = int(train_args.get("latent_dim", 128))
    image_size = int(train_args.get("image_size", 128))
    base_channels = int(train_args.get("base_channels", 64))

    g = Generator(
        latent_dim=latent_dim,
        image_size=image_size,
        base_channels=base_channels,
    ).to(device)

    state_key = "G" if use_live else "G_ema"
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint is missing '{state_key}' weights.")
    g.load_state_dict(ckpt[state_key])
    g.eval()

    return g, {
        "latent_dim": latent_dim,
        "image_size": image_size,
        "base_channels": base_channels,
        "epoch": int(ckpt.get("epoch", 0)),
        "state_key": state_key,
    }


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    G, meta = load_generator(checkpoint_path, device, use_live=args.use_live)
    latent_dim = meta["latent_dim"]
    print(
        f"Loaded {meta['state_key']} from epoch {meta['epoch']} "
        f"(image_size={meta['image_size']}, latent_dim={latent_dim})",
        flush=True,
    )

    # Decide which latents to generate. Interpolation mode lays each
    # interpolation out as one row in the saved grid so the morph reads
    # left to right.
    if args.interpolate:
        if args.interpolate_steps < 2 or args.interpolate_pairs < 1:
            raise ValueError(
                "interpolate-steps must be >= 2 and interpolate-pairs >= 1"
            )
        all_latents = build_interpolation_latents(
            args.interpolate_pairs,
            args.interpolate_steps,
            latent_dim,
            device,
        )
        grid_cols = args.interpolate_steps
        total = all_latents.size(0)
    else:
        total = args.num_samples
        all_latents = sample_truncated(total, latent_dim, device, args.truncation)
        grid_cols = args.grid_cols

    # Generate in batches and render incrementally so memory stays bounded.
    produced = 0
    chunks: list[torch.Tensor] = []
    while produced < total:
        bs = min(args.batch_size, total - produced)
        z = all_latents[produced : produced + bs]
        with torch.no_grad():
            samples = G(z).clamp(-1, 1)
            samples = (samples + 1) / 2  # map from [neg one, one] to [zero, one]
        samples_cpu = samples.cpu()
        chunks.append(samples_cpu)

        if args.cards:
            render_batch(samples_cpu, output_dir, prefix="card", start_index=produced)

        produced += bs

    all_samples = torch.cat(chunks, dim=0)
    grid_path = output_dir / ("interpolation.png" if args.interpolate else "grid.png")
    save_image(all_samples, grid_path, nrow=grid_cols)

    mode = "interpolation" if args.interpolate else "samples"
    print(
        f"Wrote {all_samples.size(0)} {mode} to {output_dir} (grid: {grid_path})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
