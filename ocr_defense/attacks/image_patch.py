from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ..render import FreeTypeRenderer, RenderConfig


@dataclass(frozen=True)
class ImagePatchAttackConfig:
    # How many patches to try per rendered line.
    max_patches_per_line: int = 1
    # Fraction of line bbox width/height to use for patch rectangles.
    patch_width_ratio: Tuple[float, float] = (0.2, 0.6)
    patch_height_ratio: Tuple[float, float] = (0.2, 0.7)
    # Pixel noise parameters.
    pixel_value_min: int = 40
    pixel_value_max: int = 220
    pixel_fill_mode: str = "random"  # "random" | "constant"
    # Effects to combine.
    effects: Tuple[str, ...] = ("bbox", "pixel", "text")
    # Small text used for the "text" effect.
    text_charset: str = string.ascii_uppercase + string.digits
    patch_text_length_range: Tuple[int, int] = (3, 8)
    # Font size for patch text; if None uses renderer font_size//2.
    patch_font_size: Optional[int] = None
    random_seed: Optional[int] = None


def _clamp(v: int, a: int, b: int) -> int:
    return max(a, min(b, v))


def _composite_or(base: Image.Image, overlay: Image.Image, bbox: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    x1 = _clamp(x1, 0, base.width - 1)
    y1 = _clamp(y1, 0, base.height - 1)
    x2 = _clamp(x2, 0, base.width)
    y2 = _clamp(y2, 0, base.height)
    if x2 <= x1 or y2 <= y1:
        return
    for y in range(y1, y2):
        for x in range(x1, x2):
            ov = overlay.getpixel((x, y))
            if isinstance(ov, tuple):
                if any(ov):
                    bv = base.getpixel((x, y))
                    base.putpixel((x, y), tuple(max(int(bv[i]), int(ov[i])) for i in range(len(ov))))
            else:
                if ov:
                    base.putpixel((x, y), max(base.getpixel((x, y)), ov))


def image_patch_attack(
    image: Image.Image,
    *,
    renderer: FreeTypeRenderer,
    line_bboxes: Sequence[Tuple[int, int, int, int]],
    config: ImagePatchAttackConfig,
) -> Image.Image:
    """
    Image-level coding attack (UDUP-like, simplified):
    overlay visual patches per text line using bbox/pixel/text effects.
    """
    rng = random.Random(config.random_seed)
    img = image.copy()

    for (bx1, by1, bx2, by2) in line_bboxes:
        line_w = max(1, bx2 - bx1)
        line_h = max(1, by2 - by1)

        if config.max_patches_per_line <= 0:
            patches = 0
        else:
            # Attacking mode: ensure at least one patch per line.
            patches = rng.randint(1, config.max_patches_per_line)
        for _ in range(patches):
            w_ratio = rng.uniform(*config.patch_width_ratio)
            h_ratio = rng.uniform(*config.patch_height_ratio)
            pw = max(1, int(line_w * w_ratio))
            ph = max(1, int(line_h * h_ratio))

            px1 = rng.randint(bx1, max(bx1, bx2 - pw))
            py1 = rng.randint(by1, max(by1, by2 - ph))
            px2 = min(img.width, px1 + pw)
            py2 = min(img.height, py1 + ph)
            patch_bbox = (px1, py1, px2, py2)

            if "bbox" in config.effects:
                draw = ImageDraw.Draw(img)
                # Draw a filled rectangle with light noise color.
                fill_val = rng.randint(config.pixel_value_min, config.pixel_value_max)
                if img.mode == "RGB":
                    draw.rectangle(patch_bbox, fill=(fill_val, fill_val, fill_val))
                else:
                    draw.rectangle(patch_bbox, fill=fill_val)

            if "pixel" in config.effects:
                draw = ImageDraw.Draw(img)
                # Fill rectangle with per-pixel noise (kept small by patch sizes).
                for y in range(py1, py2):
                    for x in range(px1, px2):
                        if config.pixel_fill_mode == "constant":
                            v = int((config.pixel_value_min + config.pixel_value_max) / 2)
                        else:
                            v = rng.randint(config.pixel_value_min, config.pixel_value_max)
                        v = min(255, max(0, v))
                        if img.mode == "RGB":
                            img.putpixel((x, y), (v, v, v))
                        else:
                            img.putpixel((x, y), v)

            if "text" in config.effects:
                # Render a small string inside the patch bbox.
                font_size = config.patch_font_size
                if font_size is None:
                    font_size = max(8, renderer.config.font_size // 2)

                patch_text_len = rng.randint(*config.patch_text_length_range)
                patch_text = "".join(rng.choice(config.text_charset) for _ in range(patch_text_len))

                patch_cfg = RenderConfig(
                    image_width=img.width,
                    image_height=img.height,
                    font_path=renderer.config.font_path,
                    font_size=font_size,
                    dpi=renderer.config.dpi,
                    # Патч-текст должен быть видимым; рендерим чёрным по белому.
                    text_color="#000000",
                    background_color="#FFFFFF",
                )
                with FreeTypeRenderer(patch_cfg) as small_renderer:
                    overlay = small_renderer.render(patch_text, x=px1, y=py1, line_spacing=None, record_line_bboxes=False)
                _composite_or(img, overlay, patch_bbox)

    return img

