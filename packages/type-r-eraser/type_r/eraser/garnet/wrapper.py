import numpy as np
import torch

from ..base import BaseTextEraser
from .garnet import GaRNet


class GarnetWrapper(BaseTextEraser):
    def __init__(
        self,
        weight_path: str,
        polygon_dilation: bool,
        dilate_kernel_size: int,
        dilate_iteration: int,
        erase_all: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            polygon_dilation=polygon_dilation,
            dilate_kernel_size=dilate_kernel_size,
            dilate_iteration=dilate_iteration,
            erase_all=erase_all,
            *args,
            **kwargs,
        )
        self.model = GaRNet(3)
        self.device = "cuda:0"
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)

    def get_inpainted_img(self, img: np.ndarray, mask: np.ndarray):
        box_mask = mask[np.newaxis, :, :]
        im = img.transpose(2, 0, 1).astype(np.float32)
        im = im / 127.5 - 1
        im, box_mask = torch.FloatTensor(im), torch.FloatTensor(box_mask)
        x = torch.cat([im, box_mask], axis=0).unsqueeze(0).to(self.device)
        # inference
        with torch.no_grad():
            _, _, _, _, result, _, _, _ = self.model(x)

        # save result
        result = (1 - box_mask) * im + box_mask * result.cpu()
        img = (
            (torch.clamp(result[0] + 1, 0, 2) * 127.5)
            .cpu()
            .detach()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )
        return np.array(img)
