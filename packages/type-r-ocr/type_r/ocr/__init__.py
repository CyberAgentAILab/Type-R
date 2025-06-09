from typing import Any

from .base import BaseOCRDet, BaseOCRRecog
from .clip4str import Clip4Str
from .clovarecog import ClovaRecogBench
from .craft import craftOCRDet
from .deepsolo import DeepSoloDet
from .hisam import HiSAMOCRDet
from .masktextspotterv3 import (
    MaskTextSpotterV3Det,
    MaskTextSpotterV3Recog,
)
from .modelscopeocr import ConvnextTinyOCR
from .paddleocr import PaddleOCRDet
from .trocr import TROCRRecog

OCR_DETECTION_MODEL_FACTORY = {
    "paddleocr": PaddleOCRDet,
    "craft": craftOCRDet,
    "hisam": HiSAMOCRDet,
    "deepsolo": DeepSoloDet,
    "masktextspotterv3": MaskTextSpotterV3Det,
}
OCR_RECOGNITION_MODEL_FACTORY = {
    "modelscope": ConvnextTinyOCR,  # ppv3?
    "masktextspotterv3": MaskTextSpotterV3Recog,
    "trocr": TROCRRecog,
    "clovarecog": ClovaRecogBench,
    "clip4str": Clip4Str,
}


def build_detection_ocr(
    detection_model_name: str,
    detection_model_params: Any = None,
) -> BaseOCRDet:
    detection_model = OCR_DETECTION_MODEL_FACTORY[detection_model_name](
        **detection_model_params
    )
    return detection_model


def build_recognition_ocr(
    recognition_model_name: str,
    recognition_model_params: Any = None,
) -> BaseOCRRecog:
    recognition_model = OCR_RECOGNITION_MODEL_FACTORY[recognition_model_name](
        **recognition_model_params
    )
    return recognition_model
