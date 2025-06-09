import base64
import json
import os
from io import BytesIO
from typing import Any

import openai
from PIL import Image

from .base import BaseT2I


class GPTImage(BaseT2I):
    """
    https://learn.microsoft.com/ja-jp/azure/ai-services/openai/dall-e-quickstart?tabs=dalle3%2Ccommand-line&pivots=programming-language-python
    """

    def __init__(
        self,
        max_num_trials: int = 5,
        quality: str = "low",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        api_version = "2025-04-01-preview"
        self._client = openai.AzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("AZURE_OPENAI_IMAGE_API_KEY"),
            azure_endpoint=os.environ["AZURE_OPENAI_IMAGE_ENDPOINT"],
        )
        self.max_num_trials = max_num_trials
        self.quality = quality

    def sample(self, prompt: str, height: int, width: int) -> Image.Image:
        n_retry = 0
        while n_retry < self.max_num_trials:
            try:
                response = self._client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",  # (1024x1024、1792x1024、1024x1792)
                    quality=self.quality,
                    # response_format="b64_json",  # url or b64_json
                    # quality="hd",  # (standard, hd)
                    # style="natural",  # (vivid or natural)
                )
            except openai.BadRequestError as e:
                print(e)
                n_retry += 1
                continue
            break

        if n_retry == self.max_num_trials:
            print(f"{n_retry=} reached the maximum retry count {self.max_num_trials=}.")
            image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        else:
            json_response = json.loads(response.model_dump_json())
            image_bytes = base64.b64decode(json_response["data"][0]["b64_json"])
            image = Image.open(BytesIO(image_bytes))

        image = image.resize((width, height))
        return image
