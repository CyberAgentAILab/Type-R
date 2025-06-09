import os

import hydra
import numpy as np
import torch
from hydra.utils import call, instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from type_r.adjuster.wordchain import wordchain, wordchain_mapping
from type_r.ocr.base import BaseOCRDet, BaseOCRRecog
from type_r.util.cv_func import draw_pos, min_bounding_rect
from type_r.util.gen_util import extract_words_from_prompt
from type_r.util.structure import Polygon, WordMapping


def get_wordchain(
    words_prompt: list[str],
    words_ocr: list[str],
) -> dict[str, str]:
    """
    Generate a mapping between words from a prompt and words recognized by OCR, along with the associated costs.
    Args:
        words_prompt (list[str]): List of words from the prompt.
        words_ocr (list[str]): List of words recognized by OCR.
    Returns:
        prompt2ocr (dict): Mapping from prompt words to OCR words.
    """
    src_indexes, tgt_indexes = wordchain(
        words_prompt,
        words_ocr,
    )
    prompt2ocr = wordchain_mapping(words_prompt, words_ocr, src_indexes, tgt_indexes)
    return prompt2ocr


def filter_polygons(
    polygons: list[Polygon],
    filtering_size_rate: float = 0.04,
    img_height: int = 512,
    img_width: int = 512,
) -> tuple[list[Polygon], list[Polygon]]:
    """
    Filters a list of polygons based on their height rate.
    Args:
        polygons (list[Polygon]): A list of Polygon objects to be filtered.
        filtering_size_rate (float, optional): The threshold rate for filtering polygons based on their height.
                                               Polygons with a height rate greater than this value will be kept.
                                               Defaults to 0.05.
    Returns:
        tuple: A tuple containing two lists:
            - The first list contains polygons that passed the filtering.
            - The second list contains polygons that were filtered out.
    """

    _polygons = []
    _polygons_filterd = []
    for i, polygon in enumerate(polygons):
        pos = (
            draw_pos(np.array(polygon), 1.0, height=img_height, width=img_width) * 255.0
        )
        tl, tr, br, bl = min_bounding_rect(pos.astype(np.uint8))
        tc = (tl + tr) / 2
        bc = (bl + br) / 2
        height = np.linalg.norm(tc - bc)
        height_rate = float(height) / img_height
        if height_rate > filtering_size_rate:
            _polygons.append(polygon)
        else:
            _polygons_filterd.append(polygon)
    return _polygons, _polygons_filterd


def get_word_mapping(
    prompt: str,
    image: np.ndarray,
    ocr_detection: BaseOCRDet,
    ocr_recognition: BaseOCRRecog,
    filtering_size_rate: float = 0.04,
    filter: bool = False,
):
    """
    Generates a mapping between words in a prompt and words detected in an image using OCR.
    Args:
        prompt (str): The input text prompt containing words to be mapped.
        image (np.array): The image array on which OCR detection and recognition are performed.
        ocr_detection (BaseOCRDet): An OCR detection model to detect text regions in the image.
        ocr_recognition (BaseOCRRecog): An OCR recognition model to recognize text within detected regions.
        filtering_size_rate (float): The rate used to filter out small polygons during OCR detection.
        filter (bool): A flag indicating whether to filter detected polygons based on size. Defaults to False.
    Returns:
        WordMapping: An object containing the mapping between words in the prompt and words detected in the image.
    """
    ocr_outs = ocr_detection(image)
    polygons = ocr_outs.polygons
    if filter:
        polygons, polygons_filterd = filter_polygons(polygons, filtering_size_rate)
        words_ocr = ocr_recognition(image, polygons)
        words_prompt = extract_words_from_prompt(prompt)
        prompt2ocr = get_wordchain(words_prompt, words_ocr)
        filted_polygons_num = len(polygons_filterd)
        wordmapping = WordMapping(
            words_ocr=words_ocr + [""] * filted_polygons_num,
            words_prompt=words_prompt,
            polygons=polygons + polygons_filterd,
            prompt2ocr=prompt2ocr,
        )
    else:
        words_ocr = ocr_recognition(image, polygons)
        words_prompt = extract_words_from_prompt(prompt)
        prompt2ocr = get_wordchain(words_prompt, words_ocr)
        wordmapping = WordMapping(
            words_ocr=words_ocr,
            words_prompt=words_prompt,
            polygons=polygons,
            prompt2ocr=prompt2ocr,
        )
        logger.info(f"{wordmapping.words_prompt=}")
        logger.info(f"{wordmapping.words_ocr=}")
        logger.info(f"{wordmapping.prompt2ocr=}")
    return wordmapping


def load_image(
    target_dir: str, dataset: str, idx: int, img_height: int = 512, img_width: int = 512
) -> np.ndarray:
    ref_img = Image.open(os.path.join(target_dir, f"{dataset}_{str(idx)}.jpg"))
    ref_img = np.array(ref_img.resize((img_width, img_height))).astype(np.uint8)
    return ref_img


def load_word_mapping(target_dir: str, dataset: str, idx: int) -> WordMapping:
    mapping = WordMapping.load_json(
        os.path.join(target_dir, f"{dataset}_{str(idx)}.json")
    )
    return mapping


@hydra.main(version_base=None, config_path="../config", config_name="layout_correction")
def layout_correction(config: DictConfig):
    ##########################
    # I/O
    ##########################
    root_res = config.output_dir
    os.makedirs(os.path.join(root_res, "layout_corrected_img"), exist_ok=True)
    os.makedirs(os.path.join(root_res, "word_mapping"), exist_ok=True)

    ##########################
    # Load modules
    ##########################
    hfds = call(config.dataset)()()
    ocr_detection = instantiate(config.ocr_detection)()
    ocr_recognition = instantiate(config.ocr_recognition)()
    text_eraser = instantiate(config.text_eraser)()
    adjuster = instantiate(config.adjuster)(use_azure=config.use_azure)

    ##########################
    # main loop
    ##########################
    for i, element in enumerate(hfds):
        dataset = element["dataset_name"]
        idx = element["id"]
        prompt = element["prompt"]
        logger.info(f"{i=} {dataset}, {idx=}, {prompt=}")

        ################
        # load image
        ################
        ref_img = load_image(config.reference_img_dir, dataset, idx)

        ################
        # word mapping
        ################
        if config.load_word_mapping:
            word_mapping = load_word_mapping(config.word_mapping_dir, dataset, idx)
        else:
            word_mapping = get_word_mapping(
                prompt,
                ref_img,
                ocr_detection,
                ocr_recognition,
                filter=config.filter_detection,
                filtering_size_rate=config.filtering_size_rate,
            )

        ################
        # adjust word mapping
        ################
        logger.info(f"{word_mapping.words_prompt=}")
        logger.info(f"{word_mapping.words_ocr=}")
        logger.info(f"{word_mapping.prompt2ocr=}")
        text_num_aligned_img, word_mapping_adjusted = adjuster(
            image=ref_img, word_mapping=word_mapping, text_eraser=text_eraser
        )
        logger.info(f"{word_mapping_adjusted.words_prompt=}")
        logger.info(f"{word_mapping_adjusted.words_ocr=}")
        logger.info(f"{word_mapping_adjusted.prompt2ocr=}")

        save_name = os.path.join(
            root_res, "layout_corrected_img", f"{dataset}_{str(idx)}.jpg"
        )
        Image.fromarray(text_num_aligned_img.astype(np.uint8)).save(save_name)

        save_name = os.path.join(root_res, "word_mapping", f"{dataset}_{str(idx)}.json")
        word_mapping_adjusted.dump_json(save_name)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    logger.info("start layout correction")
    layout_correction()
