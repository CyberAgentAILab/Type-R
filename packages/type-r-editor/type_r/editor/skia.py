import random

import numpy as np
import skia

from .base import BaseTextEditor


class SkiaEditor(BaseTextEditor):
    def __init__(
        self,
        font_path: str,
        white_bg: bool = False,
        *args,
        **kwargs,
    ):
        fontmgr = skia.FontMgr()
        self.font_type_face = fontmgr.makeFromFile(font_path, 0)
        self.white_bg = white_bg

    def get_inpainted_img(self, img, target_texts, polygons, *args, **kwargs):
        def draw_fill(canvas, textblob, x, y, fill_paint):
            canvas.drawTextBlob(textblob, x, y, fill_paint)

        if len(polygons) > 0:
            surface = skia.Surface(img.shape[1], img.shape[0])

            def draw_img(canvas, img):
                tmp = (
                    np.zeros((img.shape[0], img.shape[1], 4)).astype(dtype=np.uint8)
                    + 255
                )
                tmp[:, :, 0:3] = img.copy()
                image = skia.Image.fromarray(tmp)
                canvas.drawImage(image, 0, 0)

            def check_rgb_chennel(surface, canvas, img):
                _img = surface.makeImageSnapshot().toarray()[:, :, 0:3]
                if np.all(_img == img):
                    return canvas
                else:
                    canvas = surface.getCanvas()
                    draw_img(canvas, img[:, :, ::-1])
                    return canvas

            def get_img_canvas_with_check_rgb_chennel(surface, img):
                canvas = surface.getCanvas()
                draw_img(canvas, img)
                canvas = check_rgb_chennel(surface, canvas, img)
                return canvas

            def get_text_width_bottom(font_skia, text):
                glyphs = font_skia.textToGlyphs(text)
                positions = font_skia.getPos(glyphs)
                rects = font_skia.getBounds(glyphs)
                try:
                    text_left = positions[0].x() + rects[0].left()
                    text_right = positions[-1].x() + rects[-1].right()
                    text_width = text_right - text_left
                    text_top = min(
                        [
                            position.y() + rect.top()
                            for position, rect in zip(positions, rects)
                        ]
                    )
                    text_bottom = max(
                        [
                            position.y() + rect.bottom()
                            for position, rect in zip(positions, rects)
                        ]
                    )
                    text_height = text_bottom - text_top
                except IndexError:
                    text_width = 0
                    text_height = 0
                    text_bottom = 0
                return text_width, text_height, text_bottom

            def get_random_color():
                if random.random() < 0.2:
                    val = random.randint(230, 255)
                    color = [val, val, val]
                elif random.random() < 0.4:
                    val = random.randint(0, 50)
                    color = [val, val, val]
                else:
                    color = [
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    ]
                return skia.ColorSetRGB(color[0], color[1], color[2])

            def get_black_color():
                return skia.ColorSetRGB(0, 0, 0)

            if self.white_bg:
                canvas = get_img_canvas_with_check_rgb_chennel(
                    surface, np.full_like(img, 255)
                )
            else:
                canvas = get_img_canvas_with_check_rgb_chennel(surface, img)
            for text, polygon in zip(target_texts, polygons):
                box_h = polygon[2][1] - polygon[0][1]
                font_size = round(box_h * 0.8)
                font = skia.Font(self.font_type_face, font_size, 1, 1e-20)
                textblob = skia.TextBlob(text, font)
                text_width, text_height, text_bottom = get_text_width_bottom(font, text)
                fill_paint = skia.Paint(
                    AntiAlias=True,
                    Color=get_black_color(),
                    Style=skia.Paint.kFill_Style,
                )
                fill_paint.setBlendMode(skia.BlendMode.kSrcOver)
                x = (polygon[0][0] + polygon[1][0]) // 2 - text_width // 2
                y = polygon[2][1] - (box_h - text_height) / 2 - text_bottom
                draw_fill(canvas, textblob, x, y, fill_paint)
            img = surface.makeImageSnapshot().toarray()
            return img[:, :, 0:3].astype(np.uint8)
        else:
            return img