from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sample import (  # noqa: E402
    build_interpolation_latents,
    sample_truncated,
    slerp,
)


def test_sample_truncated_no_threshold_matches_shape() -> None:
    z = sample_truncated(8, latent_dim=16, device=torch.device("cpu"), truncation=None)
    assert z.shape == (8, 16, 1, 1)


def test_sample_truncated_respects_threshold() -> None:
    z = sample_truncated(64, latent_dim=32, device=torch.device("cpu"), truncation=0.5)
    assert z.shape == (64, 32, 1, 1)
    assert z.abs().max().item() <= 0.5 + 1e-6


def test_sample_truncated_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        sample_truncated(4, latent_dim=8, device=torch.device("cpu"), truncation=0.0)


def test_slerp_endpoints_recover_inputs() -> None:
    torch.manual_seed(0)
    z0 = torch.randn(1, 8, 1, 1)
    z1 = torch.randn(1, 8, 1, 1)
    out = slerp(z0.expand(2, -1, -1, -1), z1.expand(2, -1, -1, -1), torch.tensor([0.0, 1.0]))
    assert out.shape == (2, 8, 1, 1)
    # Slerp scales the endpoints onto the great circle between the unit
    # vectors of z0 and z1, so directions must match even if magnitudes do not.
    cos0 = torch.nn.functional.cosine_similarity(
        out[0].flatten(0), z0.flatten(0), dim=0
    )
    cos1 = torch.nn.functional.cosine_similarity(
        out[1].flatten(0), z1.flatten(0), dim=0
    )
    assert cos0.item() > 0.999
    assert cos1.item() > 0.999


def test_build_interpolation_latents_shape() -> None:
    z = build_interpolation_latents(
        pairs=3, steps=5, latent_dim=12, device=torch.device("cpu")
    )
    assert z.shape == (15, 12, 1, 1)
