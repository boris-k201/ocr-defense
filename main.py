from PIL import Image
from freetype.raw import *
import argparse
from pathlib import Path
import math

font_path = Path('.') / 'fonts' / 'PT_Sans' / 'PTSans-Regular.ttf'
font_size = 16
image_size = (800, 600)

def to_c_str(text):
    ''' Convert python strings to null terminated c strings. '''
    cStr = create_string_buffer(text.encode(encoding='UTF-8'))
    return cast(pointer(cStr), POINTER(c_char))

def draw_bitmap( image, bitmap, x, y):
    x_max = x + bitmap.width
    y_max = y + bitmap.rows
    p = 0
    for p,i in enumerate(range(x,x_max)):
        for q,j in enumerate(range(y,y_max)):
            if i < 0  or j < 0 or i >= image_size[0] or j >= image_size[1]:
                continue;
            pixel = image.getpixel((i,j))
            pixel |= int(bitmap.buffer[q * bitmap.width + p]);
            image.putpixel((i,j), pixel)


def init_freetype():
    library = FT_Library()
    matrix  = FT_Matrix()
    face    = FT_Face()

    # initialize library, error handling omitted
    error = FT_Init_FreeType( byref(library) )

    # create face object, error handling omitted
    error = FT_New_Face( library, to_c_str(str(font_path)), 0, byref(face) )

    # set character size: 16pt at 300dpi, error handling omitted
    error = FT_Set_Char_Size( face, 0, font_size * 64, 96, 96 )
    slot = face.contents.glyph

    # set up matrix
    angle = 0
    matrix.xx = (int)( math.cos( angle ) * 0x10000 )
    matrix.xy = (int)(-math.sin( angle ) * 0x10000 )
    matrix.yx = (int)( math.sin( angle ) * 0x10000 )
    matrix.yy = (int)( math.cos( angle ) * 0x10000 )

    return library, matrix, face, slot

def draw_text(library, matrix, face, slot, image, text, x, y):
    pen     = FT_Vector()
    # the pen position in 26.6 cartesian space coordinates; */
    # start at (300,200) relative to the upper left corner  */
    pen.x = x * 64;
    pen.y = int((image_size[1] - face.contents.height / 64 - y) * 64)

    for i, c in enumerate(text):
        # set transformation
        FT_Set_Transform( face, byref(matrix), byref(pen) )

        # load glyph image into the slot (erase previous one)
        charcode = ord(c)
        index = FT_Get_Char_Index( face, charcode )
        FT_Load_Glyph( face, index, FT_LOAD_RENDER )

        # now, draw to our target surface (convert position)
        draw_bitmap( image, slot.contents.bitmap,
                     slot.contents.bitmap_left,
                     image_size[1] - slot.contents.bitmap_top )

        # increment pen position
        pen.x += slot.contents.advance.x
        pen.y += slot.contents.advance.y

def main():
    library, matrix, face, slot = init_freetype()
    image = Image.new('L', image_size)
    text = 'Добрый день,\n мир!'

    draw_text(library, matrix, face, slot, image, text, 0, 0)

    FT_Done_Face(face)
    FT_Done_FreeType(library)

    image.save('test.png')

if __name__ == '__main__':
    main()
