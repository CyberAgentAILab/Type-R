import logging
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import paddleocr
from paddleocr.paddleocr import (
    BASE_DIR,
    SUPPORT_DET_MODEL,
    SUPPORT_OCR_MODEL_VERSION,
    SUPPORT_REC_MODEL,
    alpha_to_color,
    binarize_img,
    check_gpu,
    check_img,
    confirm_model_dir_url,
    get_model_config,
    logger,
    maybe_download,
    parse_args,
    parse_lang,
    predict_system,
)
from PIL import Image

from ..base import BaseOCRDet, OCRDetOut


def extract_text_and_box(results):
    # Process and draw results
    texts = []
    boxes = []
    for res in results:
        if res is not None:
            for line in res:
                box = [tuple(point) for point in line[0]]
                # # Finding the bounding box
                # box = [
                #     (min(point[0] for point in box), min(point[1] for point in box)),
                # ]
                txt = line[1][0]
                texts.append(txt)
                boxes.append(box)
    return texts, boxes


def filter_ocr(
    texts: list[str], text_boxes: list[tuple[tuple[int, int], tuple[int, int]]]
) -> tuple[list[str], list[tuple[tuple[int, int], tuple[int, int]]]]:
    _texts = []
    _text_boxes = []
    for text, text_box in zip(texts, text_boxes):
        x0 = min(point[0] for point in text_box)
        y0 = min(point[1] for point in text_box)
        x1 = max(point[0] for point in text_box)
        y1 = max(point[1] for point in text_box)
        h = y1 - y0
        w = x1 - x0
        if h > 5 and w > 5:
            _texts.append(text)
            _text_boxes.append(text_box)
    return _texts, _text_boxes


# for setting det_lang = "ml"
class PaddleOCRCustom(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, (
            "ocr_version must in {}, but get {}".format(
                SUPPORT_OCR_MODEL_VERSION, params.ocr_version
            )
        )
        params.use_gpu = check_gpu(params.use_gpu)

        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)
        det_lang = "ml"  # code of lapper

        # init model dir
        det_model_config = get_model_config("OCR", params.ocr_version, "det", det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, "whl", "det", det_lang),
            det_model_config["url"],
        )
        rec_model_config = get_model_config("OCR", params.ocr_version, "rec", lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, "whl", "rec", lang),
            rec_model_config["url"],
        )
        cls_model_config = get_model_config("OCR", params.ocr_version, "cls", "ch")
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, "whl", "cls"),
            cls_model_config["url"],
        )
        if params.ocr_version in ["PP-OCRv3", "PP-OCRv4"]:
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"
        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error("det_algorithm must in {}".format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error("rec_algorithm must in {}".format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            # params.rec_char_dict_path = str(
            #     Path(__file__).parent / rec_model_config["dict_path"]
            # )
            params.rec_char_dict_path = str(
                Path(paddleocr.__file__).parent / rec_model_config["dict_path"]
            )

        # logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num
        logger.setLevel(logging.CRITICAL)

    def ocr(
        self,
        img,
        det=True,
        rec=True,
        cls=True,
        bin=False,
        inv=False,
        alpha_color=(255, 255, 255),
        slice={},
    ):
        """
        OCR with PaddleOCR

        args:
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
            slice: use sliding window inference for large images, det and rec must be True. Requires int values for slice["horizontal_stride"], slice["vertical_stride"], slice["merge_x_thres"], slice["merge_y_thres] (See doc/doc_en/slice_en.md). Default is {}.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error("When input a list of images, det must be false")
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                "Since the angle classifier is not initialized, it will not be used during the forward process"
            )

        img, _, _ = check_img(img)
        # for infer pdf file
        # if isinstance(img, list) and flag_pdf:
        #     if self.page_num > len(img) or self.page_num == 0:
        #         imgs = img
        #     else:
        #         imgs = img[: self.page_num]
        # else:
        #     imgs = [img]
        imgs = [img]

        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin:
                _image = binarize_img(_image)
            return _image

        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if dt_boxes.size == 0:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = preprocess_image(img)
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


@lru_cache(maxsize=None)
def load_paddleocr(**kwargs):
    return PaddleOCRCustom(use_angle_cls=True, lang="en")
    # return PaddleOCR(use_angle_cls=True, lang="en")


class PaddleOCRDet(BaseOCRDet):
    def __init__(self, *args, **kwargs):
        self.model = load_paddleocr()

    def inference(self, image: np.ndarray) -> OCRDetOut:
        with tempfile.NamedTemporaryFile() as fp:
            img_path = fp.name
            Image.fromarray(image).save(img_path, "JPEG")
            ocr_res = self.model.ocr(img_path)
            word, word_boxes = extract_text_and_box(ocr_res)
            _, word_boxes = filter_ocr(word, word_boxes)
        return OCRDetOut(polygons=word_boxes, modelname="paddle")
