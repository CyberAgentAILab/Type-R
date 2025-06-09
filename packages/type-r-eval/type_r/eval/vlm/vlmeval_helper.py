import logging
from typing import Any, Literal

# from langchain_core.pydantic_v1 import BaseModel, validator
from pydantic import BaseModel
from vlmeval.config import supported_VLM

from .json_util import parse_json_markdown


class BaseOutput(BaseModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def get_mock(self):
        raise NotImplementedError


logger = logging.getLogger(__name__)


TYPES = Literal["text", "image"]


def can_take_system_prompt(name) -> bool:
    if name.startswith("GPT4o"):
        return True
    else:
        return False


def parse_unknown(unknown: list[str]) -> dict:
    """
    Minimal parser for unknown arguments. It can only process single int, float, or str.
    """

    output = {}
    assert len(unknown) % 2 == 0
    for i in range(0, len(unknown), 2):
        key, value = unknown[i], unknown[i + 1]
        assert key.startswith("--")
        key = key[2:]

        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        output[key] = value

    return output


class VLMEvalWrapper:
    def __init__(
        self,
        vlm_name: str,
        output_object: BaseOutput | None = None,
        parse_retry: int = 5,  # number of retries for checking whether the output is following the schema
        system_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        logger.warning(
            f"Got {kwargs=}, which will be passed to the VLM. Please refer to the VLM's documentation."
        )

        # Some models may not take system_prompt in initialization.
        # Then, we will append it to the beginning of the message in the __call__ method.
        if can_take_system_prompt(vlm_name):
            kwargs["system_prompt"] = system_prompt
            self.system_prompt = None
        else:
            self.system_prompt = system_prompt

        # some model-specific (likely-to-be-forgotten) arguments here
        if vlm_name == "GPT4o":
            kwargs["use_azure"] = True

        self.model = supported_VLM[vlm_name](**kwargs)
        self.output_object = output_object
        self.parse_retry = parse_retry

    def __call__(self, message: list[str]) -> BaseOutput | str:
        """
        note: for message, see https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/api/base.py#L73-L92
        note: if generate_inner function supports **kwargs, this function should also support **kwargs.
        e.g.,
        - GPT4o: support `temperature` and `max_tokens`
        """
        assert isinstance(message, list)
        if self.system_prompt is not None:
            message = [self.system_prompt] + list(message)
        print(f"{message=}")

        n_retry = 0
        while True:
            if n_retry >= self.parse_retry:
                # get out of the loop by marking as failed.
                if self.output_object is not None:
                    output = None
                else:
                    output = "fake output"
                break

            content = self.model.generate(message)
            if content == "Failed to obtain answer via API.":
                output = None
                break
            if self.output_object is not None:
                try:
                    output = self.output_object.model_validate(
                        parse_json_markdown(content)
                    )
                except Exception as e:
                    logger.error(f"Failed to parse {content=} because of {e=}")
                    n_retry += 1
                    continue

            # if the script reaches here, the parsing was successful
            break

        return output
