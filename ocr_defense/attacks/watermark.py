from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from ..render import parse_rgb_color


@dataclass(frozen=True)
class WatermarkAttackConfig:
    text_lines: Tuple[str, ...] = ("CONFIDENTIAL",)
    color: Union[str, Sequence[int]] = "#606060"
    alpha: int = 80
    font_path: Optional[str] = None
    font_size: int = 24
    x_spacing: int = 160
    y_spacing: int = 120
    angle_deg: float = -18.0
    x_offset: int = 0
    y_offset: int = 0


def _load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        p = Path(font_path)
        if p.is_file():
            return ImageFont.truetype(str(p), size=font_size)
    try:
        # Generic fallback available on most systems with PIL.
        return ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except Exception:
        return ImageFont.load_default()


def watermark_attack(image: Image.Image, config: WatermarkAttackConfig) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    rgb = parse_rgb_color(config.color)
    alpha = max(0, min(255, int(config.alpha)))
    fill = (rgb[0], rgb[1], rgb[2], alpha)
    font = _load_font(config.font_path, config.font_size)

    lines = [t for t in config.text_lines if t]
    if not lines:
        lines = ["CONFIDENTIAL"]

    text = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])

    # Extend loops a bit outside the canvas so rotation still covers corners.
    for y in range(-th, base.height + th, max(20, int(config.y_spacing))):
        for x in range(-tw, base.width + tw, max(40, int(config.x_spacing))):
            draw.multiline_text(
                (x + config.x_offset, y + config.y_offset),
                text,
                fill=fill,
                font=font,
                spacing=4,
            )

    if abs(config.angle_deg) > 0.01:
        overlay = overlay.rotate(config.angle_deg, expand=False, resample=Image.BICUBIC)
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")

