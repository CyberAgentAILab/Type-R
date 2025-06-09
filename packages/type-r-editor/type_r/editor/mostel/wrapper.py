import importlib
import os
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from type_r.util.cv_func import crop_image_w_bbox, polygons2bboxes

from ..base import BaseTextEditor

# from . import standard_text
# from .model import Generator

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
    print(image.shape)
    return image.unsqueeze(0)


class MostelWrapper(BaseTextEditor):
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
        cfg = OmegaConf.load(config_yaml)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model = Generator(cfg, in_channels=3).to(device)
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint["generator"])
        model.eval()
        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.data_shape),
                transforms.ToTensor(),
            ]
        )
        self.std_text = standard_text.Std_Text(font_path)

    def get_inpainted_img(
        self,
        image: np.ndarray,
        target_texts: list[str],
        polygons: list[Polygon],
        *args,
        **kwargs,
    ):
        bboxex = polygons2bboxes(polygons)

        res_imgs = []
        for bbox, target_text in zip(bboxex, target_texts):
            source_image = crop_image_w_bbox(image, bbox)
            source_image = Image.fromarray(source_image)
            w, h = source_image.size
            i_t = self.std_text.draw_text(target_text)
            i_t = Image.fromarray(np.uint8(i_t))
            i_s = self.transform(source_image).unsqueeze(0).to(self.device)
            i_t = self.transform(i_t).unsqueeze(0).to(self.device)
            _, _, gen_o_f, _, _, _ = self.model(i_t, i_s)
            gen_o_f = gen_o_f * 255
            o_f = gen_o_f[0].detach().to("cpu").numpy().transpose(1, 2, 0)
            o_f = Image.fromarray(np.uint8(o_f)).resize((w, h))
            res_imgs.append(np.array(o_f))

        for bbox, res_img in zip(bboxex, res_imgs):
            y0, y1, x0, x1 = bbox
            image[y0:y1, x0:x1] = res_img
        return image
        return image
