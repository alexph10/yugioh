from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class CardSpec:
    """Mockup card geometry. All values are in pixels."""

    width: int = 480
    height: int = 700
    margin: int = 24
    title_height: int = 56
    art_size: int = 432
    footer_height: int = 140
    border_color: tuple[int, int, int] = (32, 32, 32)
    frame_color: tuple[int, int, int] = (200, 168, 64)
    title_color: tuple[int, int, int] = (245, 230, 180)
    footer_color: tuple[int, int, int] = (245, 230, 200)
    text_color: tuple[int, int, int] = (16, 16, 16)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try a few common system fonts, then fall back to the PIL default."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def tensor_to_pil(art: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in either [zero, one] or [neg one, one] to PIL."""
    if art.dim() == 4:
        art = art[0]
    if art.dim() != 3:
        raise ValueError(f"Expected a CHW or NCHW tensor, got shape {tuple(art.shape)}")

    art = art.detach().to(dtype=torch.float32, device="cpu")
    if float(art.min()) < 0.0:
        art = (art + 1.0) / 2.0
    art = art.clamp(0.0, 1.0)

    array = (art.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def render_card(
    art: Union[Image.Image, torch.Tensor],
    title: str = "Generated Card",
    stat_line: str = "ATK / 1000   DEF / 1000",
    description: str = "A creature dreamed up by a generative model.",
    spec: Optional[CardSpec] = None,
) -> Image.Image:
    """Composite a generated image into a stylized card mockup."""
    spec = spec or CardSpec()

    if isinstance(art, torch.Tensor):
        art = tensor_to_pil(art)

    art = art.convert("RGB").resize(
        (spec.art_size, spec.art_size), Image.Resampling.LANCZOS
    )

    canvas = Image.new("RGB", (spec.width, spec.height), spec.frame_color)
    draw = ImageDraw.Draw(canvas)

    # Outer border.
    draw.rectangle(
        [(0, 0), (spec.width - 1, spec.height - 1)],
        outline=spec.border_color,
        width=4,
    )

    # Title bar.
    title_box = (
        spec.margin,
        spec.margin,
        spec.width - spec.margin,
        spec.margin + spec.title_height,
    )
    draw.rectangle(title_box, fill=spec.title_color, outline=spec.border_color, width=2)
    title_font = _load_font(28)
    draw.text(
        (title_box[0] + 12, title_box[1] + (spec.title_height - 28) // 2),
        title,
        fill=spec.text_color,
        font=title_font,
    )

    # Art panel.
    art_x = (spec.width - spec.art_size) // 2
    art_y = spec.margin + spec.title_height + 12
    canvas.paste(art, (art_x, art_y))
    draw.rectangle(
        [
            (art_x - 2, art_y - 2),
            (art_x + spec.art_size + 1, art_y + spec.art_size + 1),
        ],
        outline=spec.border_color,
        width=2,
    )

    # Footer with stats and flavor text.
    footer_top = art_y + spec.art_size + 12
    footer_box = (
        spec.margin,
        footer_top,
        spec.width - spec.margin,
        footer_top + spec.footer_height,
    )
    draw.rectangle(
        footer_box, fill=spec.footer_color, outline=spec.border_color, width=2
    )

    stat_font = _load_font(20)
    desc_font = _load_font(16)
    draw.text(
        (footer_box[0] + 12, footer_box[1] + 10),
        stat_line,
        fill=spec.text_color,
        font=stat_font,
    )
    draw.text(
        (footer_box[0] + 12, footer_box[1] + 44),
        description,
        fill=spec.text_color,
        font=desc_font,
    )

    return canvas


def render_batch(
    arts: torch.Tensor,
    output_dir: Union[str, Path],
    titles: Optional[list[str]] = None,
    prefix: str = "card",
    start_index: int = 0,
) -> list[Path]:
    """Render a batch of generated images and save them as PNG card mockups.

    Pass `start_index` when calling repeatedly so successive batches do not
    overwrite each other on disk.
    """
    if arts.dim() != 4:
        raise ValueError(f"Expected an NCHW tensor, got shape {tuple(arts.shape)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for i in range(arts.size(0)):
        global_idx = start_index + i
        title = (
            titles[i]
            if titles and i < len(titles)
            else f"{prefix.title()} #{global_idx + 1:03d}"
        )
        card = render_card(arts[i], title=title)
        path = output_dir / f"{prefix}_{global_idx:04d}.png"
        card.save(path)
        saved.append(path)
    return saved
