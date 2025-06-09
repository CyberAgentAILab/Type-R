from typing import TypeAlias

import numpy as np
from pydantic import BaseModel

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


class OCRDetOut(BaseModel):
    """Outputs of the detection model for OCR."""

    polygons: list[Polygon]  # List of polygons [[x,y],...] for each text
    modelname: str


class BaseOCRRecog:
    def __init__(
        self,
    ) -> None:
        pass

    def inference(self, img: np.ndarray, given_layout: list[Polygon]) -> list[str]:
        raise NotImplementedError()

    def __call__(self, img: np.ndarray, given_layout: list[Polygon]) -> list[str]:
        assert type(img) is np.ndarray, "Input image should be numpy array"
        assert len(img.shape) == 3, "Input image should be RGB"
        assert img.shape[2] == 3, "Input image should be [H, W, 3]"
        assert type(given_layout) is list, "Input layout should be List of polygons"
        if len(given_layout) > 0:
            assert type(given_layout[0]) is list, (
                "The format of polygon is [[x,y],...] "
            )
            assert len(given_layout[0][0]) == 2, "The format of polygon is [[x,y],...] "
            return self.inference(img, given_layout)
        else:
            return []


class BaseOCRDet:
    def __init__(
        self,
    ) -> None:
        pass

    def inference(self, img: np.ndarray) -> OCRDetOut:
        raise NotImplementedError()

    def __call__(self, img: np.ndarray) -> OCRDetOut:
        assert type(img) is np.ndarray, "Input image should be numpy array"
        assert len(img.shape) == 3, "Input image should be RGB"
        assert img.shape[2] == 3, "Input image should be [H, W, 3]"
        return self.inference(img)
