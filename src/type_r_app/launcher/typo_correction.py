import os
import warnings

warnings.simplefilter("ignore", FutureWarning)

import hydra
import numpy as np
import torch
from hydra.utils import call, instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from pydantic import BaseModel
from type_r.editor.base import BaseTextEditor
from type_r.ocr.base import BaseOCRRecog
from type_r.util.cv_func import harmonize_mask, polygons2mask, polygons2maskdilate
from type_r.util.structure import JsonIOModel, Polygon, WordMapping


class TextEditorInputs(BaseModel):
    """inputs fot text editor."""

    target_texts: list[str]
    source_texts: list[str]
    polygons: list[Polygon]


def get_matched_prompt_ids(matched_ocr_ids: list[str], prompt2ocr: dict[int, int]):
    prompt_ids = []
    for ocr_id in matched_ocr_ids:
        for prompt_id, _ocr_id in prompt2ocr.items():
            if ocr_id == _ocr_id:
                prompt_ids.append(prompt_id)
    return prompt_ids


def get_text_editing_inputs(wordmapping: WordMapping) -> TextEditorInputs:
    matched_ocr_ids = [
        ocr_id for ocr_id in wordmapping.prompt2ocr.values() if ocr_id != -1
    ]
    polygons_matched = [
        wordmapping.polygons[matched_id] for matched_id in matched_ocr_ids
    ]
    words_ocr_matched = [
        wordmapping.words_ocr[matched_id] for matched_id in matched_ocr_ids
    ]
    matched_prompt_ids = get_matched_prompt_ids(matched_ocr_ids, wordmapping.prompt2ocr)
    words_prompt_matched = [
        wordmapping.words_prompt[matched_id] for matched_id in matched_prompt_ids
    ]

    logger.info(f"matched ocr words {words_ocr_matched}")
    logger.info(f"matched prompt words {words_prompt_matched}")
    return TextEditorInputs(
        target_texts=words_prompt_matched,
        source_texts=words_ocr_matched,
        polygons=polygons_matched,
    )


def validate_ocr_result(texts: list[str], ocr_res: list[str], errids: list):
    success_ids = []
    for i, errid in enumerate(errids):
        target_text = texts[errid]
        _ocr_res = ocr_res[i]
        logger.info(f"{target_text=} {_ocr_res=}")
        if target_text.lower() == _ocr_res.lower():
            success_ids.append(errid)
    return success_ids


def get_error_text_inputs(inpaint_text_inputs: TextEditorInputs, errids: list):
    target_texts = []
    source_texts = []
    polygons = []
    for errid in errids:
        logger.info(f"{errid=} {inpaint_text_inputs.target_texts[errid]}")
        target_texts.append(inpaint_text_inputs.target_texts[errid])
        source_texts.append(inpaint_text_inputs.source_texts[errid])
        polygons.append(inpaint_text_inputs.polygons[errid])

    return target_texts, source_texts, polygons


def update_errids(errids: list, success_ids: list):
    for _id in success_ids:
        errids.pop(errids.index(_id))
    return errids


def update_image_with_inpainted_res(
    img: np.ndarray,
    success_ids: list[int],
    polygons: list[Polygon],
    inpainted_image: np.ndarray,
):
    img_brushup = img.copy()

    polygons_paste = [polygons[i] for i in success_ids]
    mask_paste = polygons2maskdilate(img, polygons_paste)
    polygons_keep = [polygons[i] for i in range(len(polygons)) if i not in success_ids]
    mask_keep = polygons2mask(img, polygons_keep)
    img_brushup = harmonize_mask(img_brushup, inpainted_image, mask_paste, mask_keep)
    return img_brushup


class TrialStats(JsonIOModel):
    """trial stats info."""

    inpaint_text_num: int
    all_text_num: int
    trial2errnum: dict[int, int]


def brushup_all(
    prompt: str,
    image: np.ndarray,
    text_editor: BaseTextEditor,
    ocr_recognition: BaseOCRRecog,
    inpaint_text_inputs: TextEditorInputs,
    trial_num: int = 10,
    ocr_validation: bool = True,
):
    trial2errnum = {}
    if ocr_validation is False:
        errids = list(range(len(inpaint_text_inputs.target_texts)))
        target_texts, source_texts, polygons = get_error_text_inputs(
            inpaint_text_inputs, errids
        )
        inpainted_img = text_editor(
            image,
            prompt,
            target_texts,
            source_texts,
            polygons,
        )
        success_ids = list(range(len(inpaint_text_inputs.target_texts)))
        image = update_image_with_inpainted_res(
            image, success_ids, inpaint_text_inputs.polygons, inpainted_img
        )
    else:
        errids = list(range(len(inpaint_text_inputs.target_texts)))
        target_texts, source_texts, polygons = get_error_text_inputs(
            inpaint_text_inputs, errids
        )
        ocr_res = ocr_recognition(image, polygons)
        success_ids = validate_ocr_result(
            inpaint_text_inputs.target_texts, ocr_res, errids
        )
        errids = update_errids(errids, success_ids)
        for t in range(trial_num):
            logger.info(f"trial {t}")
            if len(errids) == 0:
                break
            target_texts, source_texts, polygons = get_error_text_inputs(
                inpaint_text_inputs, errids
            )
            inpainted_img = text_editor(
                image,
                prompt,
                target_texts,
                source_texts,
                polygons,
            )
            ocr_res = ocr_recognition(inpainted_img, polygons)
            success_ids = validate_ocr_result(
                inpaint_text_inputs.target_texts, ocr_res, errids
            )
            if len(success_ids) > 0:
                image = update_image_with_inpainted_res(
                    image, success_ids, inpaint_text_inputs.polygons, inpainted_img
                )
            errids = update_errids(errids, success_ids)
            trial2errnum[t] = len(errids)
    return image, trial2errnum


def load_image(
    target_dir: str, dataset: str, idx: int, img_height: int = 512, img_width: int = 512
) -> np.ndarray:
    ref_img = Image.open(os.path.join(target_dir, f"{dataset}_{str(idx)}.jpg"))
    ref_img = np.array(ref_img.resize((img_width, img_height))).astype(np.uint8)
    return ref_img


def load_word_mapping(target_dir: str, dataset: str, idx: int) -> WordMapping:
    mapping_info = WordMapping.load_json(
        os.path.join(target_dir, f"{dataset}_{str(idx)}.json")
    )
    return mapping_info


@hydra.main(version_base=None, config_path="../config", config_name="typo_correction")
def typo_correction(config: DictConfig):
    ##########################
    # I/O
    ##########################
    root_res = config.output_dir
    os.makedirs(os.path.join(root_res, "typo_corrected_img"), exist_ok=True)

    ##########################
    # Load modules
    ##########################
    hfds = call(config.dataset)()()
    ocr_recognition = instantiate(config.ocr_recognition)()
    text_editor = instantiate(config.text_editor)(font_path=config.font_path)

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
        ref_img = load_image(config.brushup_target_dir, dataset, idx)

        ################
        # word mapping
        ################
        word_mapping = load_word_mapping(config.word_mapping_dir, dataset, idx)

        ################
        # text editing
        ################
        text_editing_inputs = get_text_editing_inputs(word_mapping)
        brushup_img, _ = brushup_all(
            prompt,
            ref_img,
            text_editor,
            ocr_recognition,
            text_editing_inputs,
            trial_num=config.trial_num,
            ocr_validation=config.ocr_validation,
        )
        save_name = os.path.join(
            root_res, "typo_corrected_img", f"{dataset}_{str(idx)}.jpg"
        )
        Image.fromarray(brushup_img.astype(np.uint8)).save(save_name)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    logger.info("start typo correction")
    typo_correction()
