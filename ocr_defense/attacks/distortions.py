from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw

from ..render import parse_rgb_color


@dataclass(frozen=True)
class DistortionsAttackConfig:
    enable_skew: bool = True
    enable_rotate: bool = True
    enable_warp: bool = True
    enable_strikethrough: bool = True

    character_distort_probability: float = 0.18
    skew_degrees: float = 10.0
    rotate_degrees: float = 9.0

    warp_probability: float = 1.0
    warp_amplitude: float = 2.0
    warp_frequency: float = 0.06

    strikethrough_probability: float = 0.45
    strikethrough_width: int = 2
    strikethrough_color: Union[str, Sequence[int]] = "#202020"

    random_seed: Optional[int] = None


_WORD_RE = re.compile(r"\S+")


def _char_bboxes(line: str, bbox: Tuple[int, int, int, int]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    x1, y1, x2, y2 = bbox
    n = max(1, len(line))
    w = max(1, x2 - x1)
    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for i, ch in enumerate(line):
        cx1 = x1 + int((i / n) * w)
        cx2 = x1 + int(((i + 1) / n) * w)
        if cx2 <= cx1:
            cx2 = cx1 + 1
        out.append((ch, (cx1, y1, cx2, y2)))
    return out


def _word_bboxes(line: str, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    n = max(1, len(line))
    w = max(1, x2 - x1)
    out: List[Tuple[int, int, int, int]] = []
    for m in _WORD_RE.finditer(line):
        s, e = m.span()
        wx1 = x1 + int((s / n) * w)
        wx2 = x1 + int((e / n) * w)
        if wx2 <= wx1:
            wx2 = wx1 + 1
        out.append((wx1, y1, wx2, y2))
    return out


def _affine_char_patch(
    patch: Image.Image,
    *,
    skew_deg: float,
    rotate_deg: float,
    use_skew: bool,
    use_rotate: bool,
) -> Image.Image:
    out = patch
    if use_skew and abs(skew_deg) > 0.01:
        k = math.tan(math.radians(skew_deg))
        w, h = out.size
        new_w = max(1, int(w + abs(k) * h))
        if k >= 0:
            data = (1, k, 0, 0, 1, 0)
        else:
            data = (1, k, -k * h, 0, 1, 0)
        out = out.transform((new_w, h), Image.AFFINE, data, resample=Image.BICUBIC)
    if use_rotate and abs(rotate_deg) > 0.01:
        out = out.rotate(rotate_deg, expand=True, resample=Image.BICUBIC)
    return out


def _warp_image_horizontal(image: Image.Image, amplitude: float, frequency: float) -> Image.Image:
    src = image.convert("RGB")
    out = Image.new("RGB", src.size, (255, 255, 255))
    for y in range(src.height):
        dx = int(round(amplitude * math.sin(frequency * y)))
        row = src.crop((0, y, src.width, y + 1))
        if dx >= 0:
            out.paste(row.crop((0, 0, src.width - dx, 1)), (dx, y))
        else:
            s = -dx
            out.paste(row.crop((s, 0, src.width, 1)), (0, y))
    return out


def distortions_attack(
    image: Image.Image,
    *,
    text: str,
    line_bboxes: Sequence[Tuple[int, int, int, int]],
    config: DistortionsAttackConfig,
) -> Tuple[Image.Image, Dict]:
    rng = random.Random(config.random_seed)
    img = image.convert("RGB")
    lines = text.split("\n")
    total_chars_distorted = 0
    words_struck = 0

    # 1) Per-character skew/rotate.
    if config.enable_skew or config.enable_rotate:
        canvas = img.copy()
        for line_idx, bbox in enumerate(line_bboxes):
            if line_idx >= len(lines):
                break
            for ch, cb in _char_bboxes(lines[line_idx], bbox):
                if ch.isspace():
                    continue
                if rng.random() > config.character_distort_probability:
                    continue
                cx1, cy1, cx2, cy2 = cb
                patch = img.crop((cx1, cy1, cx2, cy2))
                skew = rng.uniform(-config.skew_degrees, config.skew_degrees) if config.enable_skew else 0.0
                rot = rng.uniform(-config.rotate_degrees, config.rotate_degrees) if config.enable_rotate else 0.0
                warped = _affine_char_patch(
                    patch,
                    skew_deg=skew,
                    rotate_deg=rot,
                    use_skew=config.enable_skew,
                    use_rotate=config.enable_rotate,
                )
                # clear original area with white-ish background blend
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((cx1, cy1, cx2, cy2), fill=(255, 255, 255))
                px = cx1 - max(0, (warped.width - (cx2 - cx1)) // 2)
                py = cy1 - max(0, (warped.height - (cy2 - cy1)) // 2)
                canvas.paste(warped, (px, py))
                total_chars_distorted += 1
        img = canvas

    # 2) Strikethrough across selected words.
    if config.enable_strikethrough:
        draw = ImageDraw.Draw(img)
        color = parse_rgb_color(config.strikethrough_color)
        for line_idx, bbox in enumerate(line_bboxes):
            if line_idx >= len(lines):
                break
            for wx1, wy1, wx2, wy2 in _word_bboxes(lines[line_idx], bbox):
                if rng.random() > config.strikethrough_probability:
                    continue
                yy = int((wy1 + wy2) / 2)
                draw.line((wx1, yy, wx2, yy), fill=color, width=max(1, int(config.strikethrough_width)))
                words_struck += 1

    # 3) Global wave warp.
    warp_used = False
    if config.enable_warp and rng.random() <= config.warp_probability:
        img = _warp_image_horizontal(img, config.warp_amplitude, config.warp_frequency)
        warp_used = True

    meta = {
        "chars_distorted": total_chars_distorted,
        "words_struck": words_struck,
        "warp_used": warp_used,
    }
    return img, meta

