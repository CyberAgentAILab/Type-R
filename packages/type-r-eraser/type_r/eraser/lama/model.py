import numpy as np
import torch
from PIL import Image

from .utils import prepare_img_and_mask


class SimpleLama:
    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)

            cur_res = Image.fromarray(cur_res)
            return cur_res


def load_lama_model(model_path):
    simple_lama = SimpleLama(model_path)
    return simple_lama
