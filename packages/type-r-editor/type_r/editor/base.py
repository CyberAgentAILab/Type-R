from typing import TypeAlias

import numpy as np

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


class BaseTextEditor:
    def __init__(self) -> None:
        pass

    def get_inpainted_img(
        self,
        image: np.ndarray,
        prompt: str,
        target_texts: list[str],
        source_texts: list[str],
        polygons: list[Polygon],
    ):
        return NotImplemented

    def __call__(
        self,
        image: np.ndarray,
        prompt: str,
        target_texts: list[str],
        source_texts: list[str],
        polygons: list[Polygon],
    ):
        image = image.astype(np.uint8)  # we assume that the image is in uint8 format
        img = self.get_inpainted_img(
            image=image,
            prompt=prompt,
            target_texts=target_texts,
            source_texts=source_texts,
            polygons=polygons,
        )
        return img
