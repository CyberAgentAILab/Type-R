from type_r.util.loader import load_class_lazily

from .base import BaseOCRDet, BaseOCRRecog


def build_detection_ocr(
    detection_model_name: str,
    detection_model_params: dict | None = None,
) -> BaseOCRDet:
    MODELS = {
        "paddleocr": "type_r.ocr.paddleocr.PaddleOCRDet",
        "craft": "type_r.ocr.craft.craftOCRDet",
        "hisam": "type_r.ocr.hisam.HiSAMOCRDet",
        "deepsolo": "type_r.ocr.deepsolo.DeepSoloDet",
        "masktextspotterv3": "type_r.ocr.masktextspotterv3.MaskTextSpotterV3Det",
    }
    detection_model = load_class_lazily(MODELS[detection_model_name])(
        **(detection_model_params or {})
    )
    return detection_model


def build_recognition_ocr(
    recognition_model_name: str,
    recognition_model_params: dict | None = None,
) -> BaseOCRRecog:
    MODELS = {
        "modelscope": "type_r.ocr.modelscopeocr.ConvnextTinyOCR",
        "masktextspotterv3": "type_r.ocr.masktextspotterv3.MaskTextSpotterV3Recog",
        "trocr": "type_r.ocr.trocr.TROCRRecog",
        "clovarecog": "type_r.ocr.clovarecog.ClovaRecogBench",
        "clip4str": "type_r.ocr.clip4str.Clip4Str",
    }
    recognition_model = load_class_lazily(MODELS[recognition_model_name])(
        **(recognition_model_params or {})
    )
    return recognition_model
