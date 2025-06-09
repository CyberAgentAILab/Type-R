import importlib
import os
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image
from type_r.util.cv_func import crop_image_w_bbox, polygons2bboxes

from ..base import BaseTextEditor

# from .MuSA.GaMuSA import GaMuSA
# from .MuSA.GaMuSA_app import text_editing

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Any, weight_path: str, font_path: str):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    config["params"]["base_config"]["scheduler_config"] = config["params"][
        "base_config"
    ]["scheduler_config"].replace("weights", weight_path)
    config["params"]["base_config"]["vae"]["pretrained"] = config["params"][
        "base_config"
    ]["vae"]["pretrained"].replace("weights", weight_path)
    config["params"]["base_config"]["text_encoder"]["params"]["ckpt_path"] = config[
        "params"
    ]["base_config"]["text_encoder"]["params"]["ckpt_path"].replace(
        "weights", weight_path
    )
    config["params"]["base_config"]["unet_pretrained"] = config["params"][
        "base_config"
    ]["unet_pretrained"].replace("weights", weight_path)
    config["params"]["base_config"]["font_path"] = font_path
    config["params"]["base_config"]["ocr_model"]["pretrained"] = config["params"][
        "base_config"
    ]["ocr_model"]["pretrained"].replace("weights", weight_path)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_state_dict(d):
    return d.get("state_dict", d)


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch

        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(
            torch.load(ckpt_path, map_location=torch.device(location))
        )
    state_dict = get_state_dict(state_dict)
    # print(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict


def create_model(config: Any, weight_path: str, font_path: str):
    model = instantiate_from_config(config.model, weight_path, font_path).cpu()
    model.load_state_dict(load_state_dict(f"{weight_path}/model.pth"), strict=False)
    model.eval()
    return model


def to_tensor_image(
    image: np.ndarray, image_height=256, image_width=256
) -> torch.Tensor:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # img = Image.open(image_path)
    # image = T.ToTensor()(T.Resize((image_height, image_width))(img.convert("RGB")))
    image = T.ToTensor()(
        T.Resize((image_height, image_width))(Image.fromarray(image).convert("RGB"))
    )
    image = image.to(device)
    return image.unsqueeze(0)


class TextCtrlWrapper(BaseTextEditor):
    def __init__(
        self,
        weight_path: str,
        font_path: str,
        starting_layer: int = 10,
        guidance_scale: float = 2.0,
        num_inference_steps: int = 50,
        **kwargs,
    ):
        config_yaml = str(Path(__file__).parent / "default.yaml")
        self.cfgs = OmegaConf.load(config_yaml)
        self.model = create_model(self.cfgs, weight_path, font_path).cuda()
        monitor_cfg = {
            "max_length": 25,
            "loss_weight": 1.0,
            "attention": "position",
            "backbone": "transformer",
            "backbone_ln": 3,
            "checkpoint": f"{weight_path}/vision_model.pth",
            "charset_path": None,
        }
        self.pipeline = GaMuSA(self.model, monitor_cfg)

        self.font_path = font_path
        self.starting_layer = starting_layer
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

    def get_inpainted_img(
        self,
        image: np.ndarray,
        target_texts: list[str],
        source_texts: list[str],
        polygons: list[Polygon],
        *args,
        **kwargs,
    ):
        bboxex = polygons2bboxes(polygons)

        GaMuSA_images = []
        for bbox, target_text, style_text in zip(bboxex, target_texts, source_texts):
            source_image = crop_image_w_bbox(image, bbox)
            style_image = source_image.copy()
            h, w = style_image.shape[:2]
            source_image = to_tensor_image(source_image)
            style_image = to_tensor_image(style_image)
            result = text_editing(
                self.pipeline,
                source_image,
                style_image,
                style_text,
                target_text,
                starting_layer=self.starting_layer,
                ddim_steps=self.num_inference_steps,
                scale=self.guidance_scale,
            )
            reconstruction_image, GaMuSA_image = result[:]
            reconstruction_image = Image.fromarray(
                (reconstruction_image * 255).astype(np.uint8)
            ).resize((w, h))
            GaMuSA_image = Image.fromarray(
                (GaMuSA_image * 255).astype(np.uint8)
            ).resize((w, h))
            GaMuSA_images.append(np.array(GaMuSA_image))
        for bbox, GaMuSA_image in zip(bboxex, GaMuSA_images):
            y0, y1, x0, x1 = bbox
            image[y0:y1, x0:x1] = GaMuSA_image
        return image
        return image
