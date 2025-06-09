import logging

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.CRITICAL, force=True)
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

from .launcher.evaluation import evaluation
from .launcher.layout_correction import layout_correction
from .launcher.prompt_augmentation import augment
from .launcher.t2i import t2i
from .launcher.typo_correction import typo_correction


@hydra.main(version_base=None, config_path="config", config_name=None)
def main(config: DictConfig):
    OmegaConf.resolve(config)
    logger.info(f"start {config.command}")
    FUNCTION_FACTORY[config.command](config)


FUNCTION_FACTORY = {
    "full": main,
    "t2i": t2i,
    "layout_correction": layout_correction,
    "typo_correction": typo_correction,
    "evaluation": evaluation,
    "prompt-augmentation": augment,
}


if __name__ == "__main__":
    main()
