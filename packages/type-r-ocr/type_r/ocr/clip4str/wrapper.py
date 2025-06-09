import string
from functools import lru_cache

import einops
import numpy as np
import torch
from modelscope.pipelines.builder import build_pipeline
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import Tasks
from PIL import Image
from torchvision import transforms as T
from type_r.util.cv_func import adjust_image, draw_pos, min_bounding_rect

from ..base import BaseOCRRecog
from .models.utils import load_from_checkpoint


def get_transform(
    img_size: tuple[int, int], rotation: int = 0, mean: float = 0.5, std: float = 0.5
):
    transforms = []
    if rotation:
        transforms.append(lambda img: img.rotate(rotation, expand=True))
    transforms.extend(
        [
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return T.Compose(transforms)


@lru_cache(maxsize=None)
def load_modelscope_ocr(ocrmodel_cache_dir: str):
    pipeline_props = {
        "type": "convnextTiny-ocr-recognition",
        "model": ocrmodel_cache_dir,
        "device": "gpu",
    }
    cfg = ConfigDict(pipeline_props)
    predictor = build_pipeline(cfg, task_name=Tasks.ocr_recognition)
    return predictor


class Clip4Str(BaseOCRRecog):
    def __init__(
        self,
        weight_path,
        sample_k: int = 5,
        sample_k2: int = 5,
        sample_total: int = 50,
        sample_prompt: str | None = None,
        *args,
        **kwargs,
    ):
        openai_meanstd = True
        mean = (0.48145466, 0.4578275, 0.40821073) if openai_meanstd else 0.5
        std = (0.26862954, 0.26130258, 0.27577711) if openai_meanstd else 0.5
        self.transform = get_transform((224, 224), rotation=0, mean=mean, std=std)
        charset_test = string.digits + string.ascii_lowercase
        self.clip_model_path = f"{weight_path}/ViT-B-16.pt"
        kwargs = {
            "charset_test": charset_test,
            "clip_pretrained": self.clip_model_path,
        }
        self.model = (
            load_from_checkpoint(f"{weight_path}/clip4str_b_plus.ckpt", **kwargs)
            .eval()
            .to("cuda")
        )
        self.sample_k = sample_k
        self.sample_k2 = sample_k2
        self.sample_total = sample_total
        self.sample_prompt = sample_prompt

    def inference(self, image: np.ndarray, polygons: list) -> list[str]:
        text_lines_pred = []
        image = torch.from_numpy(image).to(self.model.device).float()
        image = einops.rearrange(image, "h w c-> c h w")
        for polygon in polygons:  # line
            mask = (
                draw_pos(
                    np.array(polygon), 1.0, height=image.shape[1], width=image.shape[2]
                )
                * 255.0
            )
            box = min_bounding_rect(mask.astype(np.uint8))
            inp = adjust_image(box, image).data.cpu().numpy()
            inp = Image.fromarray(np.transpose(inp, (1, 2, 0)).astype(np.uint8))
            imgs = self.transform(inp).unsqueeze(0)
            text_pred = self.model.test_step(
                (imgs.to(self.model.device), None),
                -1,
                clip_model_path=self.clip_model_path,
                clip_refine=False,
                sample_K=5,
                sample_K2=5,
                sample_total=50,
                sample_prompt=None,
                alpha=0.1,
            )[0]
            text_lines_pred.append(text_pred)
        return text_lines_pred
