import einops
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from type_r.util.cv_func import adjust_image, draw_pos, min_bounding_rect

from .base import BaseOCRRecog


class TROCRRecog(BaseOCRRecog):
    def __init__(self, weight_path, *args, **kwargs):
        self.processor = TrOCRProcessor.from_pretrained(weight_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(weight_path)
        self.model.to(torch.device("cuda"))

    def inference(self, image: np.ndarray, polygons: list) -> list[str]:
        words = []
        image = torch.from_numpy(image).float()
        image = einops.rearrange(image, "h w c-> c h w")
        for polygon in polygons:
            mask = (
                draw_pos(
                    np.array(polygon), 1.0, height=image.shape[1], width=image.shape[2]
                )
                * 255.0
            )
            box = min_bounding_rect(mask.astype(np.uint8))
            inp = (
                adjust_image(box, image)
                .data.numpy()
                .astype(np.uint8)
                .transpose((1, 2, 0))
            )
            inp = Image.fromarray(inp)
            pixel_values = (
                self.processor(images=inp, return_tensors="pt")
                .pixel_values.to(self.model.device)
                .float()
            )
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            words.append(generated_text)

        return words
