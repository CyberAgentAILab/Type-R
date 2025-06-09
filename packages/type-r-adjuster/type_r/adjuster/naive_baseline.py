from typing import Any


import numpy as np
import numpy.typing as npt
from type_r.eraser.base import BaseTextEraser
from type_r.util.structure import WordMapping

from .default import get_text_eraser_inputs


def get_all_text_erased_word_mapping(wordmapping: WordMapping):
    prompt2ocr = wordmapping.prompt2ocr
    return WordMapping(
        words_ocr=[],
        words_prompt=wordmapping.words_prompt,
        polygons=[],
        prompt2ocr={k: -1 for k in prompt2ocr.keys()},
    )


def get_word_mapping_to_erase_all_texts(wordmapping: WordMapping):
    prompt2ocr = wordmapping.prompt2ocr
    return WordMapping(
        words_ocr=wordmapping.words_ocr,
        words_prompt=wordmapping.words_prompt,
        polygons=wordmapping.polygons,
        prompt2ocr={k: -1 for k in prompt2ocr.keys()},
    )


def cut_all_words(
    image: npt.NDArray[np.uint8],
    word_mapping: WordMapping,
    text_eraser: BaseTextEraser,
) -> tuple[np.array, WordMapping]:
    word_mapping_to_erase = get_word_mapping_to_erase_all_texts(word_mapping)
    text_eraser_inputs = get_text_eraser_inputs(word_mapping_to_erase)
    word_mapping_erased = get_all_text_erased_word_mapping(word_mapping)
    text_num_aligned_img = text_eraser(
        image, text_eraser_inputs.polygons_erase, text_eraser_inputs.polygons_keep
    )
    return text_num_aligned_img, word_mapping_erased


def get_polygon(
    texts: list[str],
    font_size: int = 50,
    img_height: int = 512,
    img_width: int = 512,
    text_space_ratio: float = 0.9,
    margin_ratio: float = 0.01,
):
    word_num = len(texts)
    maxlinesize = (
        img_height * text_space_ratio - img_height * margin_ratio * word_num
    ) / word_num

    if maxlinesize < font_size:
        font_size = maxlinesize
    polygons = []
    for i, text in enumerate(texts):
        twidth = font_size * len(text)
        y0 = round(
            img_height * (1 - text_space_ratio)
            + i * font_size
            + i * img_height * margin_ratio
        )
        y1 = round(y0 + font_size)
        x0 = round(img_width // 2 - 0.5 * twidth)
        x1 = round(x0 + twidth)
        bbox = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        polygons.append(bbox)
    return polygons


def get_template_polygon_word_mapping(wordmapping: WordMapping):
    polygon = get_polygon(wordmapping.words_prompt)
    return WordMapping(
        words_ocr=wordmapping.words_prompt,
        words_prompt=wordmapping.words_prompt,
        polygons=polygon,
        prompt2ocr={k: k for k in wordmapping.prompt2ocr.keys()},
    )


class NaiveBaselineAdjuster:
    def __init__(
        self,
        skip_text_erasing: bool = True,
        **kwargs: Any,
    ) -> None:
        self.skip_text_erasing = skip_text_erasing

    def add_missing_words(
        self, image: npt.NDArray[np.uint8], word_mapping: WordMapping
    ) -> tuple[npt.NDArray[np.uint8], WordMapping]:
        word_mapping = get_template_polygon_word_mapping(word_mapping)
        return image, word_mapping

    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        word_mapping: WordMapping,
        text_eraser: BaseTextEraser,
    ) -> tuple[npt.NDArray[np.uint8], WordMapping]:
        word_mapping_matched = word_mapping
        text_num_aligned_img = image

        text_num_aligned_img, word_mapping_matched = cut_all_words(
            image=text_num_aligned_img,
            word_mapping=word_mapping_matched,
            text_eraser=text_eraser,
        )
        if -1 in word_mapping_matched.prompt2ocr.values():
            text_num_aligned_img, word_mapping_matched = self.add_missing_words(
                text_num_aligned_img, word_mapping_matched
            )
        return text_num_aligned_img, word_mapping_matched
