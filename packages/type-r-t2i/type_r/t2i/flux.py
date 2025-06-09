import os
from typing import Any

import torch
from diffusers import FluxPipeline
from PIL import Image

from .base import BaseT2I


class Flux(BaseT2I):
    # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
    def __init__(
        self,
        pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
        cache_dir: str | None = None,
        offload_state_dict: bool = False,
        low_cpu_mem_usage: bool = True,
        enable_model_cpu_offload: bool = False,
        run_on_low_vram_gpus: bool = False,  # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
        seed: int | None = None,
        image_type: str = "square",
        resolution: float = 1.0,
        resolution_type: str = "area",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            seed=seed,
            image_type=image_type,
            resolution=resolution,
            resolution_type=resolution_type,
        )
        if os.path.exists(cache_dir):
            cache_dir = self.find_cache_dir(cache_dir)
            pretrained_model_name_or_path = cache_dir

        if "schnell" in pretrained_model_name_or_path:
            # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#timestep-distilled
            self._guidance_scale = 0.0
            self._num_inference_steps = 4
            self._max_sequence_length = 256
        elif "dev" in pretrained_model_name_or_path:
            # https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux#guidance-distilled
            self._guidance_scale = 7.5
            self._num_inference_steps = 30
            self._max_sequence_length = -1
        else:
            raise NotImplementedError

        pipeline_kwargs = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "torch_dtype": torch.bfloat16,
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "offload_folder": cache_dir,
            "offload_state_dict": offload_state_dict,
        }

        pipeline = FluxPipeline.from_pretrained(**pipeline_kwargs)
        self.to_cuda_with_memory_options(
            pipeline,
            run_on_low_vram_gpus=run_on_low_vram_gpus,
            enable_model_cpu_offload=enable_model_cpu_offload,
        )
        self._pipeline = pipeline

    @property
    def sampling_kwargs(self) -> dict[str, Any]:
        return {
            "guidance_scale": self._guidance_scale,
            "num_inference_steps": self._num_inference_steps,
            "generator": self.generator,
        }

    def sample(self, prompt: str, height: int, width: int) -> Image.Image:
        if self._max_sequence_length == -1:
            sampling_kwargs = {"height": height, "width": width, **self.sampling_kwargs}
        else:
            sampling_kwargs = {
                "height": height,
                "width": width,
                "max_sequence_length": self._max_sequence_length,
                **self.sampling_kwargs,
            }
        image = self._pipeline(prompt=prompt, **sampling_kwargs).images[0]
        return image
