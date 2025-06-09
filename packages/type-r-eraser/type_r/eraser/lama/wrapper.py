import numpy as np

from ..base import BaseTextEraser
from .model import load_lama_model


class LamaWrapper(BaseTextEraser):
    def __init__(
        self,
        weight_path: str,
        polygon_dilation: bool,
        dilate_kernel_size: int,
        dilate_iteration: int,
        erase_all: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            polygon_dilation=polygon_dilation,
            dilate_kernel_size=dilate_kernel_size,
            dilate_iteration=dilate_iteration,
            erase_all=erase_all,
            *args,
            **kwargs,
        )
        self.model = load_lama_model(weight_path)

    def get_inpainted_img(self, img: np.ndarray, mask: np.ndarray):
        mask = mask[:, :, np.newaxis] * 255
        result = self.model(img, mask)
        return np.array(result)
