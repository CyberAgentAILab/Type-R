import logging
import os

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from type_r.eval.vlm.util import validate_path
from type_r.eval.vlm.vlmeval_helper import VLMEvalWrapper

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(config: DictConfig) -> None:
    print(config.dataset)
    dataset = instantiate(config.dataset)()
    tester = instantiate(config.evaluation)()

    model = VLMEvalWrapper(
        vlm_name=config.vlm_name,
        output_object=tester.pydantic_object,
        system_prompt=tester.system_prompt,
        # **parse_unknown(unknown),  # TODO: how to feed vlm-specific arguments?
    )
    output_path = validate_path(config.output_path)

    results: list[dict] = []
    for input_ in dataset:
        result = tester(model=model, input_=input_)
        results.append(result)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(str(output_path), index=False)


if __name__ == "__main__":
    main()
