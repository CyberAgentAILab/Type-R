import importlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data.dataloader import default_collate

from type_r.util.cv_func import polygons2bboxes

from ..base import BaseTextEditor
from .modules.diffusionmodules.sampling import EulerEDMSampler

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


def get_obj_from_str(string: str, reload: bool = False, invalidate_cache: bool = True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Any, weight_path):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config["params"]["conditioner_config"]["params"]["emb_models"][0]["params"][
        "ckpt_path"
    ] = config["params"]["conditioner_config"]["params"]["emb_models"][0]["params"][
        "ckpt_path"
    ].replace("./checkpoints", weight_path)
    config["params"]["conditioner_config"]["params"]["emb_models"][2]["params"][
        "config"
    ]["params"]["ckpt_path"] = config["params"]["conditioner_config"]["params"][
        "emb_models"
    ][2]["params"]["config"]["params"]["ckpt_path"].replace(
        "./checkpoints", weight_path
    )
    config["params"]["first_stage_config"]["params"]["ckpt_path"] = config["params"][
        "first_stage_config"
    ]["params"]["ckpt_path"].replace("./checkpoints", weight_path)
    config["params"]["loss_fn_config"]["params"]["predictor_config"]["params"][
        "ckpt_path"
    ] = config["params"]["loss_fn_config"]["params"]["predictor_config"]["params"][
        "ckpt_path"
    ].replace("./checkpoints", weight_path)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def init_model(cfgs: Any, weight_path: str):
    model_config_yaml = str(Path(__file__).parent / "model.yaml")
    model_cfg = OmegaConf.load(model_config_yaml)
    ckpt = f"{weight_path}/udifftext.ckpt"
    model = instantiate_from_config(model_cfg.model, weight_path)
    model.init_from_ckpt(ckpt)

    model.to(torch.device("cuda", index=cfgs.gpu))
    model.eval()
    model.freeze()

    return model


def init_sampling(cfgs: Any):
    discretization_config = {
        "target": "type_r.editor.udifftext.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
    }

    guider_config = {
        "target": "type_r.editor.udifftext.modules.diffusionmodules.guiders.VanillaCFG",
        "params": {"scale": cfgs.scale[0]},
    }

    sampler = EulerEDMSampler(
        num_steps=cfgs.steps,
        discretization_config=discretization_config,
        guider_config=guider_config,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=999.0,
        s_noise=1.0,
        verbose=True,
        device=torch.device("cuda", index=cfgs.gpu),
    )

    return sampler


def deep_copy(batch):
    c_batch = {}
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            c_batch[key] = torch.clone(batch[key])
        elif isinstance(batch[key], (tuple, list)):
            c_batch[key] = batch[key].copy()
        else:
            c_batch[key] = batch[key]

    return c_batch


def prepare_batch(cfgs: Any, batch: Any):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(torch.device("cuda", index=cfgs.gpu))

    batch_uc = deep_copy(batch)

    if "ntxt" in batch:
        batch_uc["txt"] = batch["ntxt"]
    else:
        batch_uc["txt"] = ["" for _ in range(len(batch["txt"]))]

    if "label" in batch:
        batch_uc["label"] = ["" for _ in range(len(batch["label"]))]

    return batch, batch_uc


def region_draw_text(
    H: int,
    W: int,
    r_bbox,
    text: str,
    font_path: str,  # arial.ttf
):
    m_top, m_bottom, m_left, m_right = r_bbox
    m_h, m_w = m_bottom - m_top, m_right - m_left

    font = ImageFont.truetype(font_path, 128)
    std_l, std_t, std_r, std_b = font.getbbox(text)
    std_h, std_w = std_b - std_t, std_r - std_l
    image = Image.new("RGB", (std_w, std_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, fill=(0, 0, 0), font=font, anchor="lt")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (m_h, m_w), transforms.InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.ToTensor(),
        ]
    )

    image = transform(image)

    result = torch.ones((3, H, W))
    result[:, m_top:m_bottom, m_left:m_right] = image

    return result


def predict(cfgs: Any, model: Any, sampler: Any, batch: Any):
    context = nullcontext if cfgs.aae_enabled else torch.no_grad

    with context():
        batch, batch_uc_1 = prepare_batch(cfgs, batch)

        c, uc_1 = model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc_1,
            force_uc_zero_embeddings=cfgs.force_uc_zero_embeddings,
        )

        x = sampler.get_init_noise(cfgs, model, cond=c, batch=batch, uc=uc_1)
        samples_z = sampler(
            model,
            x,
            cond=c,
            batch=batch,
            uc=uc_1,
            init_step=0,
            aae_enabled=cfgs.aae_enabled,
            detailed=cfgs.detailed,
        )

        samples_x = model.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        return samples


def augment_area(image: np.ndarray, bbox: tuple, H: int, W: int, mask_min_ratio: float):
    h, w, _ = image.shape
    m_top, m_bottom, m_left, m_right = bbox  # y0,y1,x0,x1

    mask = np.ones((h, w), dtype=np.uint8)
    mask[m_top:m_bottom, m_left:m_right] = 0

    if h >= w:
        delta = (h - w) // 2
        m_left += delta
        m_right += delta
        image = cv2.copyMakeBorder(image, 0, 0, delta, delta, cv2.BORDER_REPLICATE)
        mask = cv2.copyMakeBorder(
            mask, 0, 0, delta, delta, cv2.BORDER_CONSTANT, value=(1, 1, 1)
        )
    else:
        delta = (w - h) // 2
        m_top += delta
        m_bottom += delta
        image = cv2.copyMakeBorder(image, delta, delta, 0, 0, cv2.BORDER_REPLICATE)
        mask = cv2.copyMakeBorder(
            mask, delta, delta, 0, 0, cv2.BORDER_CONSTANT, value=(1, 1, 1)
        )

    m_h, m_w = int(m_bottom - m_top), int(m_right - m_left)
    c_h, c_w = m_top + m_h // 2, m_left + m_w // 2

    h, w, _ = image.shape
    area = (m_bottom - m_top) * (m_right - m_left)
    aug_min_ratio = mask_min_ratio * 4
    if area / (h * w) < aug_min_ratio:
        d = int((area / aug_min_ratio) ** 0.5)
        d = max(d, max(m_h, m_w))
        if c_h <= h - c_h:
            delta_top = min(c_h, d // 2)
            delta_bottom = d - delta_top
        else:
            delta_bottom = min(h - c_h, d // 2)
            delta_top = d - delta_bottom
        if c_w <= w - c_w:
            delta_left = min(c_w, d // 2)
            delta_right = d - delta_left
        else:
            delta_right = min(w - c_w, d // 2)
            delta_left = d - delta_right

        n_top, n_bottom = c_h - delta_top, c_h + delta_bottom
        n_left, n_right = c_w - delta_left, c_w + delta_right

        image = image[n_top:n_bottom, n_left:n_right, :]
        mask = mask[n_top:n_bottom, n_left:n_right]

        m_top -= n_top
        m_bottom -= n_top
        m_left -= n_left
        m_right -= n_left
    else:
        n_top = 0
        n_bottom = h
        n_left = 0
        n_right = w

    h, w, _ = image.shape
    m_top, m_bottom = int(m_top * (H / h)), int(m_bottom * (H / h))
    m_left, m_right = int(m_left * (W / w)), int(m_right * (W / w))

    image = cv2.resize(image, (W, H))
    mask = cv2.resize(mask, (W, H))

    def norm(v, m):
        return max(0, min(v, m))

    m_top = norm(m_top, H)
    m_bottom = norm(m_bottom, H)
    m_left = norm(m_left, W)
    m_right = norm(m_right, W)

    r_bbox = torch.tensor((m_top, m_bottom, m_left, m_right))
    offsets = (n_top, n_bottom, n_left, n_right)

    return image, mask, r_bbox, offsets


def get_batch(
    image: np.ndarray,
    text: str,
    bbox: tuple,
    font_path: str,
    seq_len: int,
    H: int,
    W: int,
    mask_min_ratio: float,
):
    assert len(bbox) == 4
    w, h = image.shape[1], image.shape[0]
    image, mask, r_bbox, offsets = augment_area(image, bbox, H, W, mask_min_ratio)

    image = (
        torch.from_numpy(image.transpose(2, 0, 1)).to(dtype=torch.float32) / 127.5 - 1.0
    )

    mask = torch.from_numpy(mask[None]).to(dtype=torch.float32)
    masked = image * mask
    mask = 1 - mask
    if len(text) > seq_len:
        seg_mask = torch.ones(seq_len)
        text = text[:seq_len]
    else:
        seg_mask = torch.cat((torch.ones(len(text)), torch.zeros(seq_len - len(text))))
    rendered = region_draw_text(H, W, r_bbox, text, font_path)

    # additional cond
    txt = f'"{text}"'
    original_size_as_tuple = torch.tensor((h, w))
    crop_coords_top_left = torch.tensor((0, 0))
    target_size_as_tuple = torch.tensor((H, W))

    batch = {
        "image": image,
        "mask": mask,
        "masked": masked,
        "seg_mask": seg_mask,
        "r_bbox": r_bbox,
        "rendered": rendered,
        "label": text,
        "txt": txt,
        "original_size_as_tuple": original_size_as_tuple,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size_as_tuple": target_size_as_tuple,
        "offsets": offsets,
    }
    return batch


class UdiffTextWrapper(BaseTextEditor):
    def __init__(
        self,
        weight_path: str,
        font_path: str,
        seq_len: int = 12,
        H: int = 512,
        W: int = 512,
        mask_min_ratio: float = 0.01,
        **kwargs,
    ):
        config_yaml = str(Path(__file__).parent / "default.yaml")
        self.cfgs = OmegaConf.load(config_yaml)
        self.model = init_model(self.cfgs, weight_path)
        self.sampler = init_sampling(self.cfgs)
        self.font_path = font_path
        self.seq_len = seq_len
        self.H = H
        self.W = W
        self.mask_min_ratio = mask_min_ratio

    def get_inpainted_img(
        self,
        image: np.ndarray,
        target_texts: list[str],
        polygons: list[Polygon],
        *args,
        **kwargs,
    ):
        bboxex = polygons2bboxes(polygons)
        batches = []
        for bbox, text in zip(bboxex, target_texts):
            batch = get_batch(
                image,
                text,
                bbox,
                self.font_path,
                self.seq_len,
                self.H,
                self.W,
                self.mask_min_ratio,
            )
            batches.append(batch)
        for batch in batches:
            batches_torch = default_collate([batch])
            outs = predict(self.cfgs, self.model, self.sampler, batches_torch)
            out = outs[0].data.cpu().numpy() * 255
            n_top, n_bottom, n_left, n_right = batch["offsets"]
            h = n_bottom - n_top
            w = n_right - n_left
            out = Image.fromarray(out.transpose((1, 2, 0)).astype(np.uint8)).resize(
                (w, h)
            )
            image[n_top:n_bottom, n_left:n_right] = np.array(out)
        return image
