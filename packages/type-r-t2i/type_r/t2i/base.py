import math
import os
import random
from typing import Any

import torch
from loguru import logger
from PIL import Image


class SizeSampler:
    aspect_ratios: list[float] = [
        1 / 1,
        16 / 9,
        9 / 16,
    ]

    def __init__(
        self,
        image_type: str = "arbitrary",
        resolution: float = 1.0,
        resolution_type: str = "area",
        weights: list[float] | None = None,
    ) -> None:
        assert image_type in ["arbitrary", "square"], image_type
        assert resolution_type in ["area", "pixel"], resolution_type
        assert resolution > 0.0, resolution

        if weights is not None:
            assert isinstance(weights, list) and len(weights) == len(
                SizeSampler.aspect_ratios
            ), weights
        self._weights = weights

        if image_type == "arbitrary":
            self._aspect_ratios = SizeSampler.aspect_ratios
        elif image_type == "square":
            self._aspect_ratios = [
                1 / 1,
            ]
        else:
            raise NotImplementedError
        self._image_type = image_type

        self._resolution = resolution
        self._resolution_type = resolution_type

    def __call__(self) -> tuple[int, int]:
        aspect_ratio = random.choices(self._aspect_ratios, weights=self._weights, k=1)[
            0
        ]
        W, H = SizeSampler.calculate_size_by_pixel_area(aspect_ratio, self._resolution)
        return W, H

    @staticmethod
    def calculate_size_by_pixel_area(
        aspect_ratio: float, megapixels: float
    ) -> tuple[int, int]:
        """
        https://github.com/bghira/SimpleTuner/blob/main/helpers/multiaspect/image.py#L359-L371
        """
        assert aspect_ratio > 0.0, aspect_ratio
        pixels = int(megapixels * (1024**2))

        W_new = int(round(math.sqrt(pixels * aspect_ratio)))
        H_new = int(round(math.sqrt(pixels / aspect_ratio)))

        W_new = SizeSampler.round_to_nearest_multiple(W_new, 64)
        H_new = SizeSampler.round_to_nearest_multiple(H_new, 64)

        return W_new, H_new

    @staticmethod
    def round_to_nearest_multiple(value: int, multiple: int) -> int:
        """
        Round a value to the nearest multiple.
        https://github.com/bghira/SimpleTuner/blob/main/helpers/multiaspect/image.py#L264-L268
        """
        rounded = round(value / multiple) * multiple
        return max(rounded, multiple)  # Ensure it's at least the value of 'multiple'


class BaseT2I:
    def __init__(
        self,
        use_negative_prompts: bool = False,
        seed: int | None = None,
        image_type: str = "square",
        resolution: float = 1.0,
        resolution_type: str = "area",
    ) -> None:
        if seed is None:
            seed = random.randint(0, 2**32)

        if torch.cuda.is_available():
            self._generator = torch.manual_seed(seed)
        else:
            self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._size_sampler = SizeSampler(
            image_type=image_type,
            resolution=resolution,
            resolution_type=resolution_type,
        )

    @property
    def generator(self) -> torch.Generator:
        return self._generator

    def to_cuda_with_memory_options(
        self,
        pipeline: Any,
        run_on_low_vram_gpus: bool = False,
        enable_model_cpu_offload: bool = False,
    ):
        if run_on_low_vram_gpus:
            pipeline.enable_sequential_cpu_offload()
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        elif enable_model_cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            logger.info("to cuda")
            pipeline.to("cuda")
            logger.info("to cuda done")

    def __call__(self, prompt: str) -> Image.Image:
        width, height = self._size_sampler()
        logger.info(f"{prompt=}")
        return self.sample(prompt, height=height, width=width)

    def sample(self, prompt: str, height: int, width: int) -> Image.Image:
        raise NotImplementedError

    def find_cache_dir(self, cache_dir):
        def find_model_index_dirs(root_path):
            matched_dirs = []
            for dirpath, _, filenames in os.walk(root_path):
                if "model_index.json" in filenames:
                    matched_dirs.append(dirpath)
            return matched_dirs

        matched_dirs = find_model_index_dirs(cache_dir)
        if len(matched_dirs) == 0:
            raise ValueError(f"cache_dir {cache_dir} does not contain model_index.json")
        if len(matched_dirs) > 0:
            cache_dir = matched_dirs[0]
        return cache_dir
