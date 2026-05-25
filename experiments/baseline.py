from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from augment import diff_augment  # noqa: E402
from data import YugiohCardDataset  # noqa: E402
from model import Discriminator, Generator  # noqa: E402
from render import render_batch  # noqa: E402
from train import hinge_d_loss, hinge_g_loss, update_ema  # noqa: E402


def main() -> int:
    """Smoke test the full pipeline on a small subset to confirm everything wires up."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 64
    latent_dim = 64
    batch_size = 16
    steps = 30
    base_channels = 32

    # Prefer the cropped art panels if they exist so the smoke run mirrors
    # the real training data path. Fall back to the raw card scans otherwise.
    art_dir = REPO_ROOT / "data" / "processed" / "art"
    art_manifest = art_dir / "manifest.csv"
    if art_dir.is_dir() and any(art_dir.glob("*.png")):
        images_dir = art_dir
        manifest_csv = art_manifest if art_manifest.is_file() else None
    else:
        images_dir = REPO_ROOT / "data" / "raw" / "yugioh_cards"
        manifest_csv = images_dir / "manifest.csv"

    dataset = YugiohCardDataset(
        images_dir=images_dir,
        manifest_csv=manifest_csv,
        image_size=image_size,
    )
    subset = Subset(dataset, list(range(min(256, len(dataset)))))
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )

    G = Generator(
        latent_dim=latent_dim, image_size=image_size, base_channels=base_channels
    ).to(device)
    D = Discriminator(image_size=image_size, base_channels=base_channels).to(device)
    G_ema = Generator(
        latent_dim=latent_dim, image_size=image_size, base_channels=base_channels
    ).to(device)
    G_ema.load_state_dict(G.state_dict())
    G_ema.eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    print(
        f"device={device} dataset={len(dataset)} subset={len(subset)} batches={len(loader)}",
        flush=True,
    )

    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            real = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            real = next(iterator)
        real = real.to(device)
        bs = real.size(0)

        # D step with DiffAugment on both real and fake.
        opt_d.zero_grad(set_to_none=True)
        z = torch.randn(bs, latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake = G(z)
        d_loss = hinge_d_loss(
            D(diff_augment(real)),
            D(diff_augment(fake)),
        )
        d_loss.backward()
        opt_d.step()

        # G step with DiffAugment on the generated batch.
        opt_g.zero_grad(set_to_none=True)
        z = torch.randn(bs, latent_dim, 1, 1, device=device)
        g_loss = hinge_g_loss(D(diff_augment(G(z))))
        g_loss.backward()
        opt_g.step()

        update_ema(G_ema, G, decay=0.99)

        if step % 5 == 0 or step == steps:
            print(
                f"step {step:>3d}/{steps} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}",
                flush=True,
            )

    # Render a few mockups from the EMA generator to verify the rendering path.
    out_dir = REPO_ROOT / "experiments" / "baseline_out"
    with torch.no_grad():
        samples = G_ema(torch.randn(4, latent_dim, 1, 1, device=device)).cpu()
    saved = render_batch(samples, out_dir, prefix="baseline")

    print(f"Baseline pipeline OK. Saved {len(saved)} mockups under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
