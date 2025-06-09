import os
from importlib import resources as impresources

import numpy as np
import pytest
from type_r.ocr import OCR_DETECTION_MODEL_FACTORY, OCR_RECOGNITION_MODEL_FACTORY
from type_r.util.cv_func import draw_pos

import type_r_app
from type_r_app.launcher.layout_correction import get_word_mapping


def check_polygons(polygons, expected):
    for i, (polygon, polygon_expected) in enumerate(zip(polygons, expected)):
        if i == 0:
            mask = draw_pos(np.array(polygon), 1.0) * 255.0
            mask_sample = draw_pos(np.array(polygon_expected), 1.0) * 255.0
        else:
            mask += draw_pos(np.array(polygon), 1.0) * 255.0
            mask_sample += draw_pos(np.array(polygon_expected), 1.0) * 255.0
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    mask_sample = np.clip(mask_sample, 0, 255).astype(np.uint8)
    assert np.allclose(mask, mask_sample, atol=1), "Detection result is not correct"


def check_word_list_matching(words, expected, err_msg):
    pred_sorted = sorted([word.lower() for word in words])
    gt_sorted = sorted([word.lower() for word in expected])
    assert "".join(pred_sorted) == "".join(gt_sorted), err_msg


def check_word_mapping(wordmappinginfo, expected):
    matched_ocr_ids = [
        ocr_id for ocr_id in wordmappinginfo.prompt2ocr.values() if ocr_id != -1
    ]
    ocr_words = [
        wordmappinginfo.words_ocr[matched_id] for matched_id in matched_ocr_ids
    ]
    check_word_list_matching(ocr_words, expected, "Recognition result is not correct")


@pytest.mark.gpu
def test_paddle(
    sample_image,
    sample_paddle_polygons,
    paddle,
):
    outs = paddle(sample_image)
    check_polygons(outs.polygons, sample_paddle_polygons)


@pytest.mark.skip
@pytest.mark.gpu
def test_masktextspotterv3_detection(
    sample_image,
    sample_masktextspotterv3_polygons,
    masktextspotterv3_detection,
):
    outs = masktextspotterv3_detection(sample_image)
    check_polygons(outs.polygons, sample_masktextspotterv3_polygons)


@pytest.mark.gpu
def test_craft(
    sample_image,
    sample_craft_polygons,
    craft,
):
    outs = craft(sample_image)
    check_polygons(outs.polygons, sample_craft_polygons)


@pytest.mark.gpu
def test_hisam(
    sample_image,
    hisam,
):
    hisam(sample_image)


@pytest.mark.skip
@pytest.mark.gpu
def test_deepsolo(
    sample_image,
    deepsolo,
):
    deepsolo(sample_image)


@pytest.mark.gpu
def test_modelscope(
    sample_prompt,
    sample_image,
    paddle,
    modelscope,
    sample_modelscope_words,
):
    wordmappinginfo = get_word_mapping(sample_prompt, sample_image, paddle, modelscope)
    check_word_mapping(wordmappinginfo, sample_modelscope_words)


@pytest.mark.gpu
def test_trocr(
    sample_prompt,
    sample_image,
    paddle,
    trocr,
    sample_trocr_words,
):
    wordmappinginfo = get_word_mapping(sample_prompt, sample_image, paddle, trocr)
    check_word_mapping(wordmappinginfo, sample_trocr_words)


@pytest.mark.gpu
def test_clovarecog(
    sample_prompt,
    sample_image,
    paddle,
    clovarecog,
    sample_clovarecog_words,
):
    wordmappinginfo = get_word_mapping(sample_prompt, sample_image, paddle, clovarecog)
    check_word_mapping(wordmappinginfo, sample_clovarecog_words)


def test_detection_prefix_matching():
    yaml_list = os.listdir(impresources.files(type_r_app) / "config" / "ocr_detection")
    yaml_prefix_list = [yaml.split(".")[0] for yaml in yaml_list]
    model_FACTORY_prefix_list = OCR_DETECTION_MODEL_FACTORY.keys()
    check_word_list_matching(
        yaml_prefix_list,
        model_FACTORY_prefix_list,
        "Detection yaml prefixes and FACTORY prefixes not matched",
    )


def test_recognition_prefix_matching():
    yaml_list = os.listdir(
        impresources.files(type_r_app) / "config" / "ocr_recognition"
    )
    yaml_prefix_list = [yaml.split(".")[0] for yaml in yaml_list]
    model_FACTORY_prefix_list = OCR_RECOGNITION_MODEL_FACTORY.keys()
    check_word_list_matching(
        yaml_prefix_list,
        model_FACTORY_prefix_list,
        "Recognition yaml prefixes and FACTORY prefixes not matched",
    )
