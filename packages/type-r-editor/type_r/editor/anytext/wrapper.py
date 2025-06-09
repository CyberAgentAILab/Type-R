import random
from pathlib import Path
from typing import Optional, TypeAlias

import einops
import numpy as np
import torch
from PIL import ImageFont
from type_r.util.cv_func import arr2tensor, draw_pos

from ..base import BaseTextEditor
from .cldm.ddim_hacked import DDIMSampler
from .cldm.model import create_model, load_state_dict
from .t3_dataset import draw_glyph, draw_glyph2, get_caption_pos

Point: TypeAlias = tuple[int, int]
Polygon: TypeAlias = list[Point]  # List of polygons [[x,y],...] for each text


def seed_everything(seed: Optional[int] = None) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_item(cur_item, font, max_lines=20, max_chars=20, PLACE_HOLDER="*"):
    item_dict = {}
    item_dict["img_name"] = cur_item["img_name"]
    item_dict["caption"] = cur_item["caption"]
    item_dict["glyphs"] = []
    item_dict["gly_line"] = []
    item_dict["positions"] = []
    item_dict["texts"] = []
    texts = cur_item.get("texts", [])
    if len(texts) > 0:
        sel_idxs = [i for i in range(len(texts))]
        if len(texts) > max_lines:
            sel_idxs = sel_idxs[:max_lines]
        pos_idxs = [cur_item["pos"][i] for i in sel_idxs]
        item_dict["caption"] = get_caption_pos(
            item_dict["caption"], pos_idxs, 0.0, PLACE_HOLDER
        )
        item_dict["polygons"] = [cur_item["polygons"][i] for i in sel_idxs]
        item_dict["texts"] = [cur_item["texts"][i][:max_chars] for i in sel_idxs]
        # glyphs
        for idx, text in enumerate(item_dict["texts"]):
            gly_line = draw_glyph(font, text)
            glyphs = draw_glyph2(font, text, item_dict["polygons"][idx], scale=2)
            item_dict["glyphs"] += [glyphs]
            item_dict["gly_line"] += [gly_line]
        # mask_pos
        for polygon in item_dict["polygons"]:
            item_dict["positions"] += [draw_pos(polygon, 1.0)]
    fill_caption = False
    if fill_caption:  # if using embedding_manager, DO NOT fill caption!
        for i in range(len(item_dict["texts"])):
            r_txt = item_dict["texts"][i]
            item_dict["caption"] = item_dict["caption"].replace(
                PLACE_HOLDER, f'"{r_txt}"', 1
            )
    # padding
    n_lines = min(len(texts), max_lines)
    item_dict["n_lines"] = n_lines
    n_pad = max_lines - n_lines
    if n_pad > 0:
        item_dict["glyphs"] += [np.zeros((512 * 2, 512 * 2, 1))] * n_pad
        item_dict["gly_line"] += [np.zeros((80, 512, 1))] * n_pad
        item_dict["positions"] += [np.zeros((512, 512, 1))] * n_pad
    return item_dict


class AnyTextWrapper(BaseTextEditor):
    def __init__(
        self,
        weight_path,
        font_path: str,
        max_lines: int = 20,
        max_chars: int = 20,
        PLACE_HOLDER: str = "*",
        a_prompt: str = "best quality, extremely detailed",
        n_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark",
        num_samples: int = 1,
        ddim_steps: int = 20,
        strength: float = 1.0,
        scale: float = 9.0,
        seed: int = -1,
        eta: float = 1.0,
        save_memory: bool = False,
        *args,
        **kwargs,
    ):
        config_yaml = str(Path(__file__).parent / "default.yaml")
        model = create_model(config_yaml).cuda()
        model.load_state_dict(
            load_state_dict(weight_path, location="cuda"), strict=False
        )
        self.model = model
        self.ddim_sampler = DDIMSampler(model)
        self.font = self.load_font(font_path)
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.PLACE_HOLDER = PLACE_HOLDER
        self.a_prompt = a_prompt
        self.n_prompt = n_prompt
        self.num_samples = num_samples
        self.ddim_steps = ddim_steps
        self.strength = strength
        self.scale = scale
        self.seed = seed
        self.eta = eta
        self.save_memory = save_memory

    def load_font(self, font_path: str, size: int = 60) -> ImageFont:
        return ImageFont.truetype(font_path, size=size)

    def get_item_dict(
        self,
        prompt,
        texts,
        polygons,
    ):
        info = {}
        info["img_name"] = None
        info["img_path"] = None
        info["caption"] = prompt
        info["polygons"] = [np.array(i).astype(np.int32) for i in polygons]
        info["texts"] = texts
        info["language"] = ["Latin"] * len(texts)
        info["pos"] = [-1] * len(texts)
        item_dict = get_item(
            info, self.font, self.max_lines, self.max_chars, self.PLACE_HOLDER
        )
        return item_dict

    def get_inpainted_img(
        self,
        image: np.ndarray,
        prompt: str,
        target_texts: list[str],
        polygons: list[Polygon],
        *args,
        **kwargs,
    ):
        item_dict = self.get_item_dict(prompt, target_texts, polygons)
        prompt = item_dict["caption"]
        n_lines = item_dict["n_lines"]
        target_texts = item_dict["texts"]
        pos_imgs = item_dict["positions"]
        glyphs = item_dict["glyphs"]
        gly_line = item_dict["gly_line"]

        with torch.no_grad():
            hint = np.sum(pos_imgs, axis=0).clip(0, 1)
            (
                H,
                W,
            ) = (512, 512)
            if self.seed == -1:
                seed = random.randint(0, 65535)
            else:
                seed = self.seed
            seed_everything(seed)
            if self.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            info = {}
            info["glyphs"] = []
            info["gly_line"] = []
            info["positions"] = []
            info["n_lines"] = [n_lines] * self.num_samples
            info["texts"] = []
            for i in range(n_lines):
                glyph = glyphs[i]
                pos = pos_imgs[i]
                gline = gly_line[i]
                info["glyphs"] += [arr2tensor(glyph, self.num_samples)]
                info["gly_line"] += [arr2tensor(gline, self.num_samples)]
                info["positions"] += [arr2tensor(pos, self.num_samples)]
                info["texts"] += [[target_texts[i]] * self.num_samples]
            # get masked_x
            if image is None:
                image = np.ones((H, W, 3)) * 127.5
            masked_img = ((image.astype(np.float32) / 127.5) - 1.0) * (1 - hint)
            masked_img = np.transpose(masked_img, (2, 0, 1))
            masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
            encoder_posterior = self.model.encode_first_stage(masked_img[None, ...])
            masked_x = self.model.get_first_stage_encoding(encoder_posterior).detach()
            info["masked_x"] = torch.cat(
                [masked_x for _ in range(self.num_samples)], dim=0
            )

            hint = arr2tensor(hint, self.num_samples)

            cond = self.model.get_learned_conditioning(
                dict(
                    c_concat=[hint],
                    c_crossattn=[[prompt + ", " + self.a_prompt] * self.num_samples],
                    text_info=info,
                )
            )
            un_cond = self.model.get_learned_conditioning(
                dict(
                    c_concat=[hint],
                    c_crossattn=[[self.n_prompt] * self.num_samples],
                    text_info=info,
                )
            )
            shape = (4, H // 8, W // 8)
            if self.save_memory:
                self.model.low_vram_shift(is_diffusing=True)
            self.model.control_scales = [self.strength] * 13
            samples, _ = self.ddim_sampler.sample(
                self.ddim_steps,
                self.num_samples,
                shape,
                cond,
                verbose=False,
                eta=self.eta,
                unconditional_guidance_scale=self.scale,
                unconditional_conditioning=un_cond,
            )
            if self.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            x_samples = self.model.decode_first_stage(samples)
            x_samples = x_samples.detach() * 127.5 + 127.5

        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c"))
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )[0]  # assume batch size is 1
        return x_samples
