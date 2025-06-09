from typing import TypeAlias

import numpy as np
import torch
from diffusers import FluxFillPipeline
from PIL import Image

from type_r.util.cv_func import polygons2maskdilate

from .base import BaseTextEditor

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


class FluxFillWrapper(BaseTextEditor):
    def __init__(
        self,
        weight_path: str,
        guidance_scale: float = 30,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        **kwargs,
    ):
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        self.pipe = FluxFillPipeline.from_pretrained(
            weight_path, torch_dtype=torch.bfloat16
        ).to("cuda")

    def get_inpainted_img(
        self,
        image: np.ndarray,
        polygons: list[Polygon],
        *args,
        **kwargs,
    ):
        mask = polygons2maskdilate(image, polygons) * 255
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image = self.pipe(
            prompt="a white paper cup",
            image=image,
            mask_image=mask,
            height=image.size[1],
            width=image.size[0],
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            # generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        return image
