import os
import time
from io import BytesIO
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from .base import BaseT2I


class GeminiImage(BaseT2I):
    """
    https://learn.microsoft.com/ja-jp/azure/ai-services/openai/dall-e-quickstart?tabs=dalle3%2Ccommand-line&pivots=programming-language-python
    """

    def __init__(
        self,
        max_num_trials: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # https://aistudio.google.com/app/apikey
        self.client = genai.Client(api_key=os.environ["GENAI_API_KEY"])
        self.max_num_trials = max_num_trials

    def sample(self, prompt: str, height: int, width: int) -> Image.Image:
        n_retry = 0
        while n_retry < self.max_num_trials:
            try:
                response = self.client.models.generate_content(
                    model="models/gemini-2.0-flash-exp",
                    contents=(prompt),
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"]
                    ),
                )
            except genai.errors.APIError as e:
                print(e)
                time.sleep(5)  # Wait a bit before retrying
                n_retry += 1
                continue
            try:
                imgflag = False
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        imgflag = True
                        break
                if imgflag is False:
                    time.sleep(5)  # Wait a bit before retrying
                    continue
            except Exception as e:
                print(f"Error processing response: {e}")
                time.sleep(5)
                continue
            break

        if n_retry == self.max_num_trials:
            print(f"{n_retry=} reached the maximum retry count {self.max_num_trials=}.")
            image = Image.new("RGB", (1024, 1024), (255, 255, 255))
        else:
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
                elif part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    break
        image = image.resize((width, height))
        return image
