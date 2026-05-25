"""Differentiable augmentation for GAN training.

A faithful PyTorch port of DiffAugment (Zhao et al., 2020,
https://arxiv.org/abs/2006.10738). The same random transform is applied to
both real and fake batches inside the discriminator loss, so gradients can
still flow through the augmentation pipeline.

Typical use:

    from augment import diff_augment

    d_loss = hinge_d_loss(D(diff_augment(real)), D(diff_augment(fake)))
    g_loss = hinge_g_loss(D(diff_augment(G(z))))
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

DEFAULT_POLICY = "color,translation,cutout"


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    shift = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    return x + shift


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.0
    return (x - mean) * factor + mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    )
    return (x - mean) * factor + mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    n, _, h, w = x.shape
    shift_x = int(w * ratio + 0.5)
    shift_y = int(h * ratio + 0.5)

    tx = torch.randint(-shift_x, shift_x + 1, (n, 1, 1), device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, (n, 1, 1), device=x.device)

    grid_b, grid_y, grid_x = torch.meshgrid(
        torch.arange(n, device=x.device),
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    grid_y = torch.clamp(grid_y + ty + 1, 0, h + 1)
    grid_x = torch.clamp(grid_x + tx + 1, 0, w + 1)

    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return (
        padded.permute(0, 2, 3, 1)
        .contiguous()[grid_b, grid_y, grid_x]
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    n, _, h, w = x.shape
    cut_h, cut_w = int(h * ratio + 0.5), int(w * ratio + 0.5)

    offset_x = torch.randint(0, w + (1 - cut_w % 2), (n, 1, 1), device=x.device)
    offset_y = torch.randint(0, h + (1 - cut_h % 2), (n, 1, 1), device=x.device)

    grid_b, grid_y, grid_x = torch.meshgrid(
        torch.arange(n, device=x.device),
        torch.arange(cut_h, device=x.device),
        torch.arange(cut_w, device=x.device),
        indexing="ij",
    )
    grid_y = torch.clamp(grid_y + offset_y - cut_h // 2, min=0, max=h - 1)
    grid_x = torch.clamp(grid_x + offset_x - cut_w // 2, min=0, max=w - 1)

    mask = torch.ones(n, h, w, dtype=x.dtype, device=x.device)
    mask[grid_b, grid_y, grid_x] = 0.0
    return x * mask.unsqueeze(1)


AUGMENT_FNS: dict[str, list[Callable[[torch.Tensor], torch.Tensor]]] = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x: torch.Tensor, policy: str = DEFAULT_POLICY) -> torch.Tensor:
    """Apply the requested chain of differentiable augmentations to ``x``.

    ``policy`` is a comma separated subset of ``color``, ``translation``, and
    ``cutout``. An empty policy returns the input unchanged so callers can
    disable augmentation by passing ``policy=""``.
    """
    if not policy:
        return x
    for name in policy.split(","):
        name = name.strip()
        if not name:
            continue
        fns = AUGMENT_FNS.get(name)
        if fns is None:
            raise ValueError(
                f"Unknown augmentation '{name}'. "
                f"Expected one of {sorted(AUGMENT_FNS)}."
            )
        for fn in fns:
            x = fn(x)
    return x.contiguous()
