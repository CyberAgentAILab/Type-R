from typing import TypeAlias

import numpy as np

from type_r.util.cv_func import (
    paste_inpainted_results,
    polygons2mask,
    polygons2maskdilate,
)

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


class BaseTextEraser:
    def __init__(
        self,
        polygon_dilation: bool,
        dilate_kernel_size: int,
        dilate_iteration: int,
        erase_all: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.polygon_dilation = polygon_dilation
        self.dilate_kernel_size = dilate_kernel_size
        self.dilate_iteration = dilate_iteration
        self.erase_all = erase_all

    def get_inpainted_img(self):
        return NotImplemented

    def __call__(
        self,
        img: np.ndarray,
        polygons_erase: list[Polygon],
        polygons_keep: list[Polygon],
    ):
        if self.erase_all:
            mask_erase = polygons2maskdilate(
                img,
                polygons_erase + polygons_keep,
                self.polygon_dilation,
                self.dilate_kernel_size,
                self.dilate_iteration,
            )
        else:
            mask_erase = polygons2maskdilate(
                img,
                polygons_erase,
                self.polygon_dilation,
                self.dilate_kernel_size,
                self.dilate_iteration,
            )
        inpainted_img = self.get_inpainted_img(img, mask_erase)
        mask_paste = polygons2maskdilate(
            img,
            polygons_erase,
            self.polygon_dilation,
            self.dilate_kernel_size,
            self.dilate_iteration,
        )
        mask_keep = polygons2mask(img, polygons_keep)
        inpainted_img = paste_inpainted_results(
            img, inpainted_img, mask_paste, mask_keep, paste_option="harmonize"
        )
        return inpainted_img.astype(np.uint8)
