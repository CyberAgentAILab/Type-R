import os
from typing import Any

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from .base import BaseT2I


class SD3(BaseT2I):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        cache_dir: str | None = None,
        offload_state_dict: bool = False,
        low_cpu_mem_usage: bool = True,
        use_negative_prompts: bool = True,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: int | None = None,
        image_type: str = "square",
        resolution: float = 1.0,
        resolution_type: str = "area",
        run_on_low_vram_gpus: bool = False,
        enable_model_cpu_offload: bool = False,
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

        self._guidance_scale = guidance_scale
        self._num_inference_steps = num_inference_steps
        self._use_negative_prompts = use_negative_prompts

        self.NEGATIVE = "deep fried watermark cropped out-of-frame low quality low res oorly drawn bad anatomy wrong anatomy extra limb missing limb floating limbs (mutated hands and fingers)1.4 disconnected limbs mutation mutated ugly disgusting blurry amputation synthetic rendering"
        pipeline_kwargs = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "torch_dtype": torch.bfloat16,
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "offload_folder": cache_dir,
            "offload_state_dict": offload_state_dict,
        }

        pipeline = StableDiffusion3Pipeline.from_pretrained(**pipeline_kwargs)
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
        if self._use_negative_prompts:
            sampling_kwargs = {
                "height": height,
                "width": width,
                "negative_prompt": self.NEGATIVE,
                **self.sampling_kwargs,
            }
        else:
            sampling_kwargs = {"height": height, "width": width, **self.sampling_kwargs}
        image = self._pipeline(prompt=prompt, **sampling_kwargs).images[0]
        return image


class SD3dot5(SD3):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3.5-large",
        cache_dir: str = None,
        offload_state_dict: bool = False,
        low_cpu_mem_usage: bool = True,
        use_negative_prompts: bool = True,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 40,
        seed: int = None,
        image_type: str = "square",
        resolution: float = 1.0,
        resolution_type: str = "area",
        run_on_low_vram_gpus: bool = False,
        enable_model_cpu_offload: bool = False,
    ) -> None:
        super().__init__(
            seed=seed,
            image_type=image_type,
            resolution=resolution,
            resolution_type=resolution_type,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            offload_state_dict=offload_state_dict,
            low_cpu_mem_usage=low_cpu_mem_usage,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            use_negative_prompts=use_negative_prompts,
            run_on_low_vram_gpus=run_on_low_vram_gpus,
            enable_model_cpu_offload=enable_model_cpu_offload,
        )
