import numpy as np

from .base import BaseTextEraser


class PadInpaintor(BaseTextEraser):
    def __init__(
        self,
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

    def get_inpainted_img(self, img, *args):
        return np.full_like(img, 255)
