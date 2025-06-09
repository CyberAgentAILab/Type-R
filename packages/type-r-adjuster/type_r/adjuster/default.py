from typing import Any


import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from type_r.eraser.base import BaseTextEraser
from type_r.util.structure import Polygons, WordMapping

from .wordchain import get_unmatched_ocr_ids


class TextEraserInputs(BaseModel):
    """inputs fot text eraser."""

    polygons_erase: Polygons
    polygons_keep: Polygons


def get_text_eraser_inputs(wordmapping: WordMapping):
    unmatched_ocr_ids = get_unmatched_ocr_ids(
        wordmapping.words_ocr, wordmapping.prompt2ocr
    )
    polygons_erase = [
        wordmapping.polygons[unmatched_id] for unmatched_id in unmatched_ocr_ids
    ]
    matched_ocr_ids = [
        ocr_id for ocr_id in wordmapping.prompt2ocr.values() if ocr_id != -1
    ]
    polygons_keep = [wordmapping.polygons[matched_id] for matched_id in matched_ocr_ids]
    return TextEraserInputs(
        polygons_erase=polygons_erase,
        polygons_keep=polygons_keep,
    )


def get_text_erased_word_mapping(wordmapping: WordMapping):
    matched_ocr_ids = [
        ocr_id for ocr_id in wordmapping.prompt2ocr.values() if ocr_id != -1
    ]
    polygons_matched = [
        wordmapping.polygons[matched_id] for matched_id in matched_ocr_ids
    ]
    words_ocr_matched = [
        wordmapping.words_ocr[matched_id] for matched_id in matched_ocr_ids
    ]
    words_prompt_matched = wordmapping.words_prompt
    prompt2ocr_matched = {}
    cnt = 0
    for k, v in wordmapping.prompt2ocr.items():
        if v in matched_ocr_ids:
            prompt2ocr_matched[k] = k - cnt
        else:
            prompt2ocr_matched[k] = -1
            cnt += 1

    return WordMapping(
        words_ocr=words_ocr_matched,
        words_prompt=words_prompt_matched,
        polygons=polygons_matched,
        prompt2ocr=prompt2ocr_matched,
    )


def cut_surplus_words(
    image: npt.NDArray[np.uint8],
    word_mapping: WordMapping,
    text_eraser: BaseTextEraser,
) -> tuple[np.array, WordMapping]:
    text_eraser_inputs = get_text_eraser_inputs(word_mapping)
    word_mapping_matched = get_text_erased_word_mapping(word_mapping)
    text_num_aligned_img = text_eraser(
        image, text_eraser_inputs.polygons_erase, text_eraser_inputs.polygons_keep
    )
    return text_num_aligned_img, word_mapping_matched


class DefaultAdjuster:
    def __init__(
        self,
        skip_text_erasing: bool = True,
        **kwargs: Any,
    ) -> None:
        self.skip_text_erasing = skip_text_erasing

    def add_missing_words(
        self, image: npt.NDArray[np.uint8], word_mapping: WordMapping
    ) -> tuple[npt.NDArray[np.uint8], WordMapping]:
        # Default method is to skip augmentation. Override this method to implement augmentation.
        return image, word_mapping

    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        word_mapping: WordMapping,
        text_eraser: BaseTextEraser,
    ) -> tuple[npt.NDArray[np.uint8], WordMapping]:
        unmatched_ocr_ids = get_unmatched_ocr_ids(
            word_mapping.words_ocr, word_mapping.prompt2ocr
        )

        word_mapping_matched = word_mapping
        text_num_aligned_img = image

        if len(unmatched_ocr_ids) > 0:
            if self.skip_text_erasing is True:
                pass
            else:
                text_num_aligned_img, word_mapping_matched = cut_surplus_words(
                    image=text_num_aligned_img,
                    word_mapping=word_mapping_matched,
                    text_eraser=text_eraser,
                )
        if -1 in word_mapping_matched.prompt2ocr.values():
            text_num_aligned_img, word_mapping_matched = self.add_missing_words(
                text_num_aligned_img, word_mapping_matched
            )
        return text_num_aligned_img, word_mapping_matched
