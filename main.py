import argparse
import json
import sys
import math
from pathlib import Path
from PIL import Image
from freetype.raw import *
from ctypes import create_string_buffer, cast, pointer, POINTER, c_char, byref

DEFAULT_CONFIG = {
    "image_width": 800,
    "image_height": 600,
    "font_path": "fonts/PT_Sans/PTSans-Regular.ttf",
    "font_size": 16,
    "dpi": 96
}

def load_config(config_path):
    config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            config.update(user_config)
    else:
        print(f"Файл конфигурации '{config_path}' не найден, используются значения по умолчанию.", file=sys.stderr)
    return config

def read_text(text_path):
    if text_path == "-":
        text = sys.stdin.read()
    else:
        input_path = Path(text_path)
        if not input_path.exists():
            print(f"Файл с текстом '{input_path}' не найден.", file=sys.stderr)
            sys.exit(1)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

def to_c_str(text):
    # Преобразует Python строку в null-terminated C строку.

    c_str = create_string_buffer(text.encode('utf-8'))
    return cast(pointer(c_str), POINTER(c_char))

def draw_bitmap(image, bitmap, x, y):
    # Рисует bitmap глифа на изображении в заданных координатах (верхний левый угол).

    width, height = image.size
    for row in range(bitmap.rows):
        for col in range(bitmap.width):
            px = x + col
            py = y + row
            if 0 <= px < width and 0 <= py < height:
                # битмап одноканальный, значение интенсивности
                value = bitmap.buffer[row * bitmap.width + col]
                if value:
                    # накладываем OR для совместимости с исходным стилем
                    image.putpixel((px, py), image.getpixel((px, py)) | value)

def init_freetype(font_path, font_size, dpi):
    # Инициализирует библиотеку FreeType и загружает шрифт.

    library = FT_Library()
    face = FT_Face()
    matrix = FT_Matrix()
    angle = 0.0
    matrix.xx = int(math.cos(angle) * 0x10000)
    matrix.xy = int(-math.sin(angle) * 0x10000)
    matrix.yx = int(math.sin(angle) * 0x10000)
    matrix.yy = int(math.cos(angle) * 0x10000)

    error = FT_Init_FreeType(byref(library))
    if error:
        raise RuntimeError("FT_Init_FreeType failed")
    error = FT_New_Face(library, to_c_str(str(font_path)), 0, byref(face))
    if error:
        raise RuntimeError(f"FT_New_Face failed: {font_path}")
    # устанавливаем размер шрифта (font_size в пунктах, dpi пикселей на дюйм)
    error = FT_Set_Char_Size(face, 0, font_size * 64, dpi, dpi)
    if error:
        raise RuntimeError("FT_Set_Char_Size failed")
    return library, matrix, face

def draw_line(library, matrix, face, image, text, x, y):
    # Рисует одну строку текста.
    
    slot = face.contents.glyph
    # метрики шрифта
    ascender = face.contents.ascender / 64   # расстояние от top до baseline
    # устанавливаем начальную позицию пера (baseline в координатах FreeType)
    pen = FT_Vector()
    pen.x = x * 64
    pen.y = int(image.height - (y + ascender)) * 64

    for ch in text:
        # преобразование (поворот) и позиция пера
        FT_Set_Transform(face, byref(matrix), byref(pen))

        # загружаем глиф
        index = FT_Get_Char_Index(face, ord(ch))
        FT_Load_Glyph(face, index, FT_LOAD_RENDER)

        # получаем битмап
        bitmap = slot.contents.bitmap

        # blit bitmap onto image
        draw_bitmap( image, slot.contents.bitmap,
                     slot.contents.bitmap_left,
                     int(image.height) - slot.contents.bitmap_top )

        # перемещаем перо вперёд
        pen.x += slot.contents.advance.x
        pen.y += slot.contents.advance.y

def render_text(image, library, matrix, face, text, x, y, line_spacing=None):
    # Рендерит многострочный текст на изображении.
    # text – строка с символами '\n'.
    # x, y – координаты верхнего левого угла первой строки.
    # line_spacing – межстрочный интервал (по умолчанию высота шрифта).
    
    if line_spacing is None:
        line_spacing = face.contents.height / 64
    lines = text.split('\n')
    for line in lines:
        draw_line(library, matrix, face, image, line, x, y)
        y += line_spacing

def main():
    parser = argparse.ArgumentParser(description="Рендеринг текста в изображение с помощью FreeType")
    parser.add_argument("--config", default="config.json",
                        help="Путь к JSON-файлу конфигурации (по умолчанию config.json)")
    parser.add_argument("--input", "-i", default="-",
                        help="Путь к файлу с текстом; '-' для чтения из stdin (по умолчанию)")
    parser.add_argument("--output", "-o", default="output.png",
                        help="Путь к выходному изображению (по умолчанию output.png)")
    args = parser.parse_args()

    # загрузка конфигурации
    config = load_config(config, Path(args.config))

    # чтение текста
    text = read_text(args.input)

    # параметры из конфига
    image_size = (config["image_width"], config["image_height"])
    font_path = Path(config["font_path"])
    if not font_path.exists():
        print(f"Файл шрифта '{font_path}' не найден.", file=sys.stderr)
        sys.exit(1)
    font_size = config["font_size"]
    dpi = config["dpi"]

    # инициализация FreeType
    library, matrix, face = init_freetype(font_path, font_size, dpi)

    # создание изображения (чёрный фон)
    image = Image.new('L', image_size, 0)

    # рендеринг текста
    render_text(image, library, matrix, face, text, 0, 0)

    # освобождение ресурсов
    FT_Done_Face(face)
    FT_Done_FreeType(library)

    # сохранение результата
    image.save(args.output)
    print(f"Изображение сохранено в {args.output}")

if __name__ == '__main__':
    main()
