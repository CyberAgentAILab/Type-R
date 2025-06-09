from pathlib import Path

import detectron2.data.transforms as T
import numpy as np
import torch
from adet.config.defaults import _C as cfg
from adet.data.augmentation import Pad
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model

from ..base import BaseOCRDet, OCRDetOut


class DeepSoloDet(BaseOCRDet):
    def __init__(self, weight_path, *args, **kwargs):
        cfgfle = str(Path(__file__).parent / "default.yaml")
        cfg.merge_from_file(cfgfle)
        self.predictor = ViTAEPredictor(cfg, weight_path)

    def inference(self, image: np.ndarray) -> OCRDetOut:
        predictions = self.predictor(image)
        bds = predictions["instances"].get_fields()["bd"].data.cpu().numpy()
        polygons = []
        for i in range(len(bds)):
            bd = bds[i]
            bd = np.hsplit(bd, 2)
            bd = np.vstack([bd[0], bd[1][::-1]])
            polygon = bd
            # words = _ctc_decode_recognition(
            #     predictions["instances"].get_fields()["recs"].data.cpu().numpy()[i]
            # )
            polygons.append(polygon.astype(np.int32).tolist())

        return OCRDetOut(
            polygons=polygons,
            modelname="deepsolo",
        )


class ViTAEPredictor:
    def __init__(self, cfg, weight_path):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(weight_path)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        # each size must be divided by 32 with no remainder for ViTAE
        self.pad = Pad(divisible_size=32)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = self.pad.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def _ctc_decode_recognition(rec):
    voc_size = 37
    CTLABELS = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    last_char = "###"
    s = ""
    for c in rec:
        c = int(c)
        if c < voc_size - 1:
            if last_char != c:
                if voc_size == 37 or voc_size == 96:
                    s += CTLABELS[c]
                    last_char = c
                else:
                    s += str(chr(CTLABELS[c]))
                    last_char = c
        else:
            last_char = "###"
    return s
    return s
