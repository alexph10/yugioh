from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    """Standard GAN preprocessing: bicubic resize, center crop, normalize to [neg one, one]."""
    return transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class YugiohCardDataset(Dataset):
    """Card image dataset backed by a manifest CSV or a directory scan."""

    def __init__(
        self,
        images_dir: str | Path,
        manifest_csv: Optional[str | Path] = None,
        image_size: int = 128,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> None:
        # Resolve to an absolute path so DataLoader workers spawned with a
        # different (or unsynced) working directory can still find the files.
        self.images_dir = Path(images_dir).resolve()
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"images_dir does not exist: {self.images_dir}")

        self.image_size = image_size
        self.transform = transform or default_transform(image_size)
        self.extensions = tuple(e.lower() for e in extensions)
        self.samples: list[Path] = self._collect_samples(manifest_csv)

        if not self.samples:
            raise RuntimeError(f"No usable images found under {self.images_dir}")

    def _collect_samples(self, manifest_csv: Optional[str | Path]) -> list[Path]:
        if manifest_csv is not None and Path(manifest_csv).is_file():
            paths: list[Path] = []
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("filename", "").strip()
                    if not filename:
                        continue
                    p = self.images_dir / filename
                    if p.is_file() and p.stat().st_size > 0:
                        paths.append(p)
            if paths:
                return paths

        # Fall back to a directory scan when no manifest is available.
        return sorted(
            p
            for p in self.images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in self.extensions
            and p.stat().st_size > 0
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Walk forward through the dataset on read failures so a single bad
        # file never crashes a training run. We catch a broad Exception so
        # decoder bugs, decompression bomb errors, transform errors, etc. all
        # fall through to the retry path; the first failure of each call is
        # surfaced via a warning so silent "all failed" cascades are visible.
        n = len(self.samples)
        first_error: Optional[BaseException] = None
        first_path: Optional[Path] = None
        for offset in range(n):
            path = self.samples[(index + offset) % n]
            try:
                with Image.open(path) as img:
                    return self.transform(img.convert("RGB"))
            except Exception as e:  # noqa: BLE001 - logged, then retried
                if first_error is None:
                    first_error = e
                    first_path = path
                continue

        detail = (
            f" first failure on {first_path}: "
            f"{type(first_error).__name__}: {first_error}"
            if first_error is not None
            else ""
        )
        warnings.warn(
            f"YugiohCardDataset: all {n} samples failed to load.{detail}",
            stacklevel=2,
        )
        raise RuntimeError(f"All samples failed to load.{detail}")
