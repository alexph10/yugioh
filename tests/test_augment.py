from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from augment import diff_augment  # noqa: E402


def _batch(n: int = 4, size: int = 32) -> torch.Tensor:
    return torch.randn(n, 3, size, size, requires_grad=True)


@pytest.mark.parametrize(
    "policy",
    ["", "color", "translation", "cutout", "color,translation,cutout"],
)
def test_diff_augment_preserves_shape_and_dtype(policy: str) -> None:
    x = _batch()
    y = diff_augment(x, policy=policy)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_diff_augment_empty_policy_is_identity() -> None:
    x = _batch()
    y = diff_augment(x, policy="")
    assert torch.equal(y, x)


def test_diff_augment_is_differentiable() -> None:
    x = _batch()
    y = diff_augment(x, policy="color,translation,cutout")
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_diff_augment_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError):
        diff_augment(_batch(), policy="nonexistent")
