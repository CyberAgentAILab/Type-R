import os
from typing import Any

import hydra
import numpy as np
from hydra.utils import call, instantiate
from loguru import logger
from omegaconf import DictConfig
from PIL import Image


def gen_img(
    prompt: str, t2i: Any, img_height: int = 512, img_width: int = 512
) -> Image.Image:
    ref_img = t2i(prompt)
    ref_img = np.array(ref_img.resize((img_width, img_height)))
    return ref_img


@hydra.main(version_base=None, config_path="../config", config_name="t2i")
def t2i(config: DictConfig):
    ##########################
    # I/O
    ##########################
    root_res = config.output_dir
    os.makedirs(os.path.join(root_res, "ref_img"), exist_ok=True)

    ##########################
    # Load modules
    ##########################
    hfds = call(config.dataset)(use_augmented_prompt=config.use_augmented_prompt)()
    t2i = instantiate(config.t2i)()
    logger.info(f"{hfds=}")

    ##########################
    # main loop
    ##########################
    for i, element in enumerate(hfds):
        dataset = element["dataset_name"]
        idx = element["id"]
        prompt = element["prompt"]
        logger.info(f"{i=} {dataset}, {idx=}, {prompt=}")
        ref_img = gen_img(
            prompt,
            t2i,
        )
        save_name = os.path.join(root_res, "ref_img", f"{dataset}_{str(idx)}.jpg")
        Image.fromarray(ref_img.astype(np.uint8)).save(save_name)


if __name__ == "__main__":
    logger.info("start text-to-image generation")
    t2i()
