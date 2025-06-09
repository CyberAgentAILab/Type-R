import logging
from functools import lru_cache

import einops
import numpy as np
import torch
from modelscope.pipelines.builder import build_pipeline
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Tasks
from type_r.util.cv_func import crop_image, draw_pos, pre_process

from .base import BaseOCRRecog


@lru_cache(maxsize=None)
def load_modelscope_ocr(ocrmodel_cache_dir: str):
    for name in logging.root.manager.loggerDict:
        if "modelscope" in name:
            log = logging.getLogger(name)
            log.setLevel(logging.CRITICAL)
            for handler in log.handlers[:]:
                log.removeHandler(handler)
    pipeline_props = {
        "type": "convnextTiny-ocr-recognition",
        "model": ocrmodel_cache_dir,
        "device": "gpu",
    }
    cfg = ConfigDict(pipeline_props)
    predictor = build_pipeline(cfg, task_name=Tasks.ocr_recognition)
    return predictor


class ConvnextTinyOCR(BaseOCRRecog):
    def __init__(self, weight_path, *args, **kwargs):
        self.model = load_modelscope_ocr(weight_path)

    def inference(self, image: np.ndarray, polygons: list) -> list[str]:
        text_lines_pred = []
        image = torch.from_numpy(image).to(self.model.device).float()
        image = einops.rearrange(image, "h w c-> c h w")
        for polygon in polygons:  # line
            pos = (
                draw_pos(
                    np.array(polygon), 1.0, height=image.shape[1], width=image.shape[2]
                )
                * 255.0
            )
            np_pos = pos.astype(np.uint8)
            x0_text = [crop_image(image, np_pos)]
            x0_text = pre_process(x0_text)
            rst = self.model(x0_text[0])
            text_pred = rst["text"][0]
            text_lines_pred.append(text_pred)
        return text_lines_pred
