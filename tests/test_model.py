from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import Discriminator, Generator, init_weights  # noqa: E402


@pytest.mark.parametrize("image_size", [64, 128, 256])
def test_generator_output_shape_and_range(image_size: int) -> None:
    g = Generator(latent_dim=64, image_size=image_size, base_channels=16)
    z = torch.randn(2, 64, 1, 1)
    out = g(z)
    assert out.shape == (2, 3, image_size, image_size)
    # Tanh output must live in [neg one, one].
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5


def test_generator_accepts_flat_latent() -> None:
    g = Generator(latent_dim=32, image_size=64, base_channels=16)
    z = torch.randn(4, 32)
    out = g(z)
    assert out.shape == (4, 3, 64, 64)


@pytest.mark.parametrize("image_size", [64, 128, 256])
def test_discriminator_output_shape(image_size: int) -> None:
    d = Discriminator(image_size=image_size, base_channels=16)
    x = torch.randn(2, 3, image_size, image_size)
    out = d(x)
    assert out.shape == (2,)
    assert out.dtype == torch.float32


def test_invalid_image_size_raises() -> None:
    with pytest.raises(ValueError):
        Generator(image_size=100)
    with pytest.raises(ValueError):
        Discriminator(image_size=100)


def test_gradient_flow_end_to_end() -> None:
    g = Generator(latent_dim=32, image_size=64, base_channels=16)
    d = Discriminator(image_size=64, base_channels=16)

    z = torch.randn(2, 32, 1, 1)
    score = d(g(z)).mean()
    score.backward()

    # Every trainable parameter should receive a gradient.
    assert all(p.grad is not None for p in g.parameters() if p.requires_grad)
    assert all(p.grad is not None for p in d.parameters() if p.requires_grad)


def test_init_weights_runs_without_error() -> None:
    g = Generator(latent_dim=16, image_size=64, base_channels=16)
    # Re applying the initializer must be idempotent and side effect free.
    g.apply(init_weights)
    out = g(torch.randn(1, 16, 1, 1))
    assert torch.isfinite(out).all()
