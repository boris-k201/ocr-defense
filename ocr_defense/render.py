from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from PIL import Image
from freetype.raw import *  # noqa: F403 - keep parity with original project
from ctypes import POINTER, byref, cast, create_string_buffer, pointer, c_char


RGBColor = Tuple[int, int, int]


@dataclass(frozen=True)
class RenderConfig:
    image_width: int = 800
    image_height: int = 600
    margin: int = 10
    # Если None — только системный шрифт; иначе этот файл используется для поддерживаемых глифов.
    font_path: Optional[str] = None
    font_size: int = 16
    dpi: int = 96
    # Цвет текста и фона. Можно задавать как "#RRGGBB" или [r,g,b].
    # По умолчанию: чёрный текст на белом фоне.
    text_color: Union[str, Sequence[int]] = "#000000"
    background_color: Union[str, Sequence[int]] = "#FFFFFF"


DEFAULT_RENDER_CONFIG = RenderConfig()


def _normalize_font_path_value(raw: object, default: Optional[str]) -> Optional[str]:
    if raw is None:
        return default
    if isinstance(raw, str):
        s = raw.strip()
        return s if s else None
    return str(raw)


def load_render_config(config_path: Path) -> RenderConfig:
    cfg = DEFAULT_RENDER_CONFIG
    if not config_path.exists():
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    fp = _normalize_font_path_value(user_cfg.get("font_path", cfg.font_path), cfg.font_path)
    return RenderConfig(
        image_width=int(user_cfg.get("image_width", cfg.image_width)),
        image_height=int(user_cfg.get("image_height", cfg.image_height)),
        margin=int(user_cfg.get("margin", cfg.margin)),
        font_path=fp,
        font_size=int(user_cfg.get("font_size", cfg.font_size)),
        dpi=int(user_cfg.get("dpi", cfg.dpi)),
        text_color=user_cfg.get("text_color", cfg.text_color),
        background_color=user_cfg.get("background_color", cfg.background_color),
    )


# Типичные пути к шрифтам в Linux (если нет fontconfig).
_SYSTEM_FONT_CANDIDATES: Tuple[str, ...] = (
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def resolve_system_font() -> Path:
    """
    Системный sans-serif через fontconfig (fc-match), иначе первый найденный кандидат.
    """
    fc = shutil.which("fc-match")
    if fc:
        try:
            proc = subprocess.run(
                [fc, "-f", "%{file}", "sans-serif"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                p = Path(proc.stdout.strip().splitlines()[0])
                if p.is_file():
                    return p
        except (OSError, subprocess.SubprocessError):
            pass
    for candidate in _SYSTEM_FONT_CANDIDATES:
        p = Path(candidate)
        if p.is_file():
            return p
    raise RuntimeError(
        "Не удалось найти системный шрифт: установите fontconfig или шрифт "
        "(например noto-fonts / ttf-dejavu) и проверьте /usr/share/fonts.",
    )


def to_c_str(text: str):
    # Convert Python string into a null-terminated C string pointer.
    c_str = create_string_buffer(text.encode("utf-8"))
    return cast(pointer(c_str), POINTER(c_char))  # type: ignore[name-defined]


def _clamp8(x: int) -> int:
    return 0 if x < 0 else 255 if x > 255 else x


def parse_rgb_color(value: Union[str, Sequence[int]]) -> RGBColor:
    """
    Поддерживает:
    - "#RRGGBB"
    - [r, g, b] или (r, g, b)
    - int (0..255) как оттенок серого
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("#") and len(s) == 7:
            r = int(s[1:3], 16)
            g = int(s[3:5], 16)
            b = int(s[5:7], 16)
            return (_clamp8(r), _clamp8(g), _clamp8(b))
        raise ValueError(f"Unsupported color string format: {value!r}. Expected '#RRGGBB'.")
    if isinstance(value, int):
        v = _clamp8(value)
        return (v, v, v)
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"Unsupported color value: {value!r}. Expected 3 components.")
    r, g, b = (int(seq[0]), int(seq[1]), int(seq[2]))
    return (_clamp8(r), _clamp8(g), _clamp8(b))


def draw_bitmap(image: Image.Image, bitmap, x: int, y: int, *, text_color: RGBColor) -> None:
    # Draw a FreeType grayscale glyph bitmap onto an RGB image using alpha blending.
    width, height = image.size
    for row in range(bitmap.rows):
        for col in range(bitmap.width):
            px = x + col
            py = y + row
            if 0 <= px < width and 0 <= py < height:
                # bitmap is single-channel intensity.
                value = bitmap.buffer[row * bitmap.width + col]
                if value:
                    a = value / 255.0
                    br, bg, bb = image.getpixel((px, py))
                    tr, tg, tb = text_color
                    nr = int(br * (1.0 - a) + tr * a)
                    ng = int(bg * (1.0 - a) + tg * a)
                    nb = int(bb * (1.0 - a) + tb * a)
                    image.putpixel((px, py), (_clamp8(nr), _clamp8(ng), _clamp8(nb)))


def _create_transform_matrix() -> FT_Matrix:
    matrix = FT_Matrix()
    angle = 0.0
    matrix.xx = int(math.cos(angle) * 0x10000)
    matrix.xy = int(-math.sin(angle) * 0x10000)
    matrix.yx = int(math.sin(angle) * 0x10000)
    matrix.yy = int(math.cos(angle) * 0x10000)
    return matrix


def _init_freetype_library() -> FT_Library:
    library = FT_Library()
    error = FT_Init_FreeType(byref(library))
    if error:
        raise RuntimeError("FT_Init_FreeType failed")
    return library


def load_ft_face(library: FT_Library, font_path: Path, font_size: int, dpi: int) -> FT_Face:
    face = FT_Face()
    error = FT_New_Face(library, to_c_str(str(font_path)), 0, byref(face))
    if error:
        raise RuntimeError(f"FT_New_Face failed: {font_path}")
    error = FT_Set_Char_Size(face, 0, font_size * 64, dpi, dpi)
    if error:
        FT_Done_Face(face)
        raise RuntimeError("FT_Set_Char_Size failed")
    return face


def _face_has_glyph(face: FT_Face, char: str) -> bool:
    # Индекс 0 — отсутствующий глиф (.notdef); поддерживаемый символ даёт ненулевой индекс.
    if not char:
        return False
    return FT_Get_Char_Index(face, ord(char)) != 0


def _select_face_for_char(user_face: Optional[FT_Face], system_face: FT_Face, ch: str) -> FT_Face:
    if user_face is not None and _face_has_glyph(user_face, ch):
        return user_face
    return system_face


def _line_ascender_px(system_face: FT_Face, user_face: Optional[FT_Face]) -> float:
    asc_sys = system_face.contents.ascender / 64.0
    asc_user = (user_face.contents.ascender / 64.0) if user_face else 0.0
    return max(asc_sys, asc_user)


def _line_height_px(system_face: FT_Face, user_face: Optional[FT_Face]) -> float:
    h_sys = system_face.contents.height / 64.0
    h_user = (user_face.contents.height / 64.0) if user_face else 0.0
    return max(h_sys, h_user)


def draw_line(
    library: FT_Library,
    matrix: FT_Matrix,
    system_face: FT_Face,
    user_face: Optional[FT_Face],
    image: Image.Image,
    text: str,
    x: int,
    y_top: float,
    *,
    text_color: RGBColor,
) -> None:
    # Draw a single line. Baseline учитывает максимальный ascender среди используемых лиц.
    ascender = _line_ascender_px(system_face, user_face)

    pen = FT_Vector()
    pen.x = x * 64
    pen.y = int(image.height - (y_top + ascender)) * 64

    for ch in text:
        face = _select_face_for_char(user_face, system_face, ch)
        slot = face.contents.glyph
        FT_Set_Transform(face, byref(matrix), byref(pen))

        index = FT_Get_Char_Index(face, ord(ch))
        FT_Load_Glyph(face, index, FT_LOAD_RENDER)

        bitmap = slot.contents.bitmap
        draw_bitmap(
            image,
            bitmap,
            slot.contents.bitmap_left,
            int(image.height) - slot.contents.bitmap_top,
            text_color=text_color,
        )

        pen.x += slot.contents.advance.x
        pen.y += slot.contents.advance.y


def measure_line_width(system_face: FT_Face, user_face: Optional[FT_Face], text: str) -> int:
    width_1_64 = 0
    for ch in text:
        face = _select_face_for_char(user_face, system_face, ch)
        slot = face.contents.glyph
        index = FT_Get_Char_Index(face, ord(ch))
        FT_Load_Glyph(face, index, FT_LOAD_RENDER)
        width_1_64 += slot.contents.advance.x
    return width_1_64 // 64

def split_text_by_line(system_face: FT_Face, user_face: Optional[FT_Face], text, max_width):
    result = []
    current_line = []
    for word in text.split(' '):
        ws = word.split('\n')
        while len(ws) > 1:
            w = ws.pop(0)
            lw = measure_line_width(system_face, user_face, ' '.join(current_line + [w]))
            if lw >= max_width:
                result.append(' '.join(current_line))
                current_line = []
            current_line.append(w)
            result.append(' '.join(current_line))
            current_line = []
        lw = measure_line_width(system_face, user_face, ' '.join(current_line + [ws[0]]))
        if lw >= max_width:
            result.append(' '.join(current_line))
            current_line = []
        current_line.append(ws[0])
    result.append(' '.join(current_line))
    return result

def render_text(
    image: Image.Image,
    library: FT_Library,
    matrix: FT_Matrix,
    system_face: FT_Face,
    user_face: Optional[FT_Face],
    text: str,
    x: int,
    y: int,
    line_spacing: Optional[float] = None,
    *,
    margin: int = 0,
    record_line_bboxes: bool = False,
    text_color: RGBColor,
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Render multi-line `text` onto `image`.

    Returns a list of line bboxes (x1, y1, x2, y2) if record_line_bboxes=True.
    """
    if line_spacing is None:
        line_spacing = _line_height_px(system_face, user_face) * 108 / 100

    lines = split_text_by_line(system_face, user_face, text, image.width - 2 * margin)
    line_bboxes: List[Tuple[int, int, int, int]] = []
    for i, line in enumerate(lines):
        y_top = y + margin + i * line_spacing
        draw_line(library, matrix, system_face, user_face, image, line, x + margin, y_top, text_color=text_color)
        if record_line_bboxes:
            width = measure_line_width(system_face, user_face, line)
            line_bboxes.append((x + margin, int(y_top + margin), 
                                x + margin + int(width), int(y_top + margin + line_spacing)))

    return line_bboxes if record_line_bboxes else None


class FreeTypeRenderer:
    def __init__(self, render_config: RenderConfig):
        self.config = render_config
        self.system_font_path = resolve_system_font()
        self.library = _init_freetype_library()
        self.matrix = _create_transform_matrix()
        self.system_face = load_ft_face(
            self.library,
            self.system_font_path,
            render_config.font_size,
            render_config.dpi,
        )
        self.user_face: Optional[FT_Face] = None
        if render_config.font_path:
            user_path = Path(render_config.font_path)
            if not user_path.is_file():
                raise FileNotFoundError(f"Font not found: {user_path}")
            self.user_face = load_ft_face(
                self.library,
                user_path,
                render_config.font_size,
                render_config.dpi,
            )
        self.text_rgb = parse_rgb_color(render_config.text_color)
        self.background_rgb = parse_rgb_color(render_config.background_color)

    def close(self) -> None:
        if self.user_face is not None:
            FT_Done_Face(self.user_face)
        FT_Done_Face(self.system_face)
        FT_Done_FreeType(self.library)

    def __enter__(self) -> "FreeTypeRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def render(
        self,
        text: str,
        *,
        x: int = 0,
        y: int = 0,
        line_spacing: Optional[float] = None,
        record_line_bboxes: bool = False,
    ):
        image = Image.new("RGB", (self.config.image_width, self.config.image_height), self.background_rgb)
        bboxes = render_text(
            image,
            self.library,
            self.matrix,
            self.system_face,
            self.user_face,
            text,
            x,
            y,
            margin=self.config.margin,
            line_spacing=line_spacing,
            record_line_bboxes=record_line_bboxes,
            text_color=self.text_rgb,
        )
        if record_line_bboxes:
            assert bboxes is not None
            return image, bboxes
        return image

