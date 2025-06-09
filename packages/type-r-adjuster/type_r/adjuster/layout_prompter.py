import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from mimetypes import guess_type
from typing import Any

import numpy as np
import numpy.typing as npt
import openai
from loguru import logger
from PIL import Image
from pydantic import Field, create_model
from type_r.util.structure import Polygon, WordMapping

from .default import DefaultAdjuster
from .json_util import parse_json_markdown


class LayoutPrompterAdjuster(DefaultAdjuster):
    def __init__(
        self,
        skip_text_erasing: bool = False,
        image_size: int = 512,
        canvas_size: int = 128,
        encode_image: bool = True,
        # openai GPT4 settings
        max_retry: int = 5,
        retry_interval: float = 1.0,
        use_azure: bool = False,
    ) -> None:
        super().__init__(skip_text_erasing)
        # note: currently assuming (left, top, width, height)
        self.image_size = image_size
        self.canvas_size = canvas_size

        pydantic_kwargs = {"word": (str, ...)}
        for key in ["width", "height"]:
            pydantic_kwargs[key] = (int, Field(ge=1, le=canvas_size))
        for key in ["left", "top"]:
            pydantic_kwargs[key] = (int, Field(ge=0, le=canvas_size - 1))
        self.class_element = create_model("Element", **pydantic_kwargs)
        self.class_layout = create_model(
            "Layout", **{"elements": (list[self.class_element], [])}
        )

        format_instructions = get_format_instructions(
            self.class_layout.model_json_schema()
        )
        examples = "\n".join(self.get_example_prompt(x) for x in LAYOUT_EXAMPLES)

        self.encode_image = encode_image
        self.system_prompt = SYSTEM_PROMPT.format(
            canvas_size=canvas_size,
            format_instructions=format_instructions,
            examples=examples,
        )

        # TODO: consider some choices
        # should we sort context / input / output?

        self.max_retry = max_retry
        self.retry_interval = retry_interval
        if use_azure:
            self.client = openai.AzureOpenAI()
            self.model_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME") or ""
        else:
            self.client = openai.OpenAI()
            self.model_name = "gpt-4o"

    def extract_layout_jsons(self, words: list[str], polygons: list[Polygon]) -> str:
        elements = []
        for word, polygon in zip(words, polygons):
            left, top, width, height = [
                round(d * (self.canvas_size / self.image_size))
                for d in polygon2ltwh(polygon)
            ]
            element = self.class_element(
                word=word, width=width, height=height, left=left, top=top
            )
            elements.append(element)

        if len(elements) == 0:
            output = json.dumps({"elements": []})
        else:
            output = json.dumps(self.class_layout(elements=elements).model_dump())
        return output

    def extract_keywords_jsons(self, words: list[str]) -> str:
        return json.dumps(words)

    def get_context_prompt(self, word_mapping: WordMapping) -> str:
        prompt = ""
        words, polygons, keywords = [], [], []
        for k, v in word_mapping.prompt2ocr.items():
            if v == -1:  # no matching
                keywords.append(word_mapping.words_prompt[k])
            else:
                words.append(word_mapping.words_ocr[v])
                polygons.append(word_mapping.polygons[v])

        logger.info(f"{keywords=}")

        prompt = INPUT_TEMPLATE.format(
            input_layout=self.extract_layout_jsons(words, polygons),
            input_keywords=self.extract_keywords_jsons(keywords),
        )
        return prompt

    def get_example_prompt(self, example: dict[str, Any]) -> str:
        prompt = EXAMPLE_TEMPLATE.format(
            input_layout=self.extract_layout_jsons(
                example["input"]["words"], example["input"]["polygons"]
            ),
            input_keywords=self.extract_keywords_jsons(example["output"]["words"]),
            output_layout=self.extract_layout_jsons(
                example["output"]["words"], example["output"]["polygons"]
            ),
        )
        return prompt

    def add_missing_words(
        self, image: npt.NDArray[np.uint8], word_mapping: WordMapping
    ) -> tuple[npt.NDArray[np.uint8], WordMapping]:
        messages = []

        content = []
        if self.encode_image:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image)},
                }
            )
        content.append(
            {
                "type": "text",
                "text": f"{self.get_context_prompt(word_mapping)} Output: ",
            }
        )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        n_retry = 0
        while n_retry < self.max_retry:
            time.sleep(self.retry_interval)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                choice = response.choices[0]
                output = choice.message.content
                if output == "":
                    if choice.finish_reason == "content_filter":
                        logger.error(
                            f"Content filter blocked the response for {messages[-1]['content'][-1]=}"
                        )
                        return image, word_mapping
                output_obj = parse_json_markdown(output)
                layout = self.class_layout.model_validate(output_obj)

                outwords = [element.word.lower() for element in layout.elements]
                kewyords = [
                    word_mapping.words_prompt[k].lower()
                    if v == -1
                    else word_mapping.words_ocr[v].lower()
                    for k, v in word_mapping.prompt2ocr.items()
                ]
                outwords_sorted = sorted(outwords)
                kewyords_sorted = sorted(kewyords)
                logger.info(f"{n_retry=} {outwords=}, {kewyords=}")
                if "".join(outwords_sorted) != "".join(kewyords_sorted):
                    n_retry += 1
                    continue

            except Exception as e:
                print(e)
                n_retry += 1
                continue

            break

        if n_retry == self.max_retry:
            logger.error(f"Failed to generate augmented prompt for {messages=}")
            return image, word_mapping
        else:
            # post processing
            for element in layout.elements:
                for index_prompt, word in enumerate(word_mapping.words_prompt):
                    if element.word != word:
                        continue

                    if word_mapping.prompt2ocr[index_prompt] != -1:
                        # error or duplicate words?
                        continue

                    word_mapping.words_ocr.append("")
                    (l, t, w, h) = [
                        int(v / self.canvas_size * self.image_size)
                        for v in [
                            element.left,
                            element.top,
                            element.width,
                            element.height,
                        ]
                    ]

                    word_mapping.polygons.append(ltwh2polygon((l, t, w, h)))
                    index_ocr = len(word_mapping.words_ocr) - 1
                    word_mapping.prompt2ocr[index_prompt] = index_ocr
                    break

        return image, word_mapping


def polygon2ltwh(polygon: Polygon) -> tuple[int, int, int, int]:
    coords_x, coords_y = [p[0] for p in polygon], [p[1] for p in polygon]
    l, r = min(coords_x), max(coords_x)
    t, b = min(coords_y), max(coords_y)
    w, h = r - l, b - t
    assert 0 <= w and 0 <= h, "invalid polygon"
    return l, t, w, h


def ltwh2polygon(ltwh: tuple[int, int, int, int]) -> Polygon:
    l, t, w, h = ltwh
    r, b = l + w, t + h
    return [(l, t), (r, t), (r, b), (l, b)]


def image_to_data_url(image: npt.NDArray[np.uint8]) -> str:
    # https://learn.microsoft.com/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest%2Csystem-assigned%2Cresource
    image_pil = Image.fromarray(image)
    buff = BytesIO()
    image_pil.save(buff, format="JPEG")

    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type("dummy.jpg")
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    base64_encoded_data = base64.b64encode(buff.getvalue()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


JSON_FORMAT_INSTRUCTIONS = """
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```
"""


def get_format_instructions(schema: dict[str, Any]) -> str:
    """Return the format instructions for the JSON output.
    Returns:
        The format instructions for the JSON output.
    """
    # Copy schema to avoid altering original Pydantic schema.
    schema = deepcopy(schema)
    # Remove extraneous fields.
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]
    # Ensure json in context is well-formed with double quotes.
    schema_str = json.dumps(reduced_schema)
    return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)


SYSTEM_PROMPT = (
    "You are an excellent autonomous AI Assistant. "
    "Please plan the layout for a list of keywords, given the image and layout information on already printed texts. "
    "Note that the canvas size is {canvas_size}x{canvas_size}.\n"
    "{format_instructions}"
    "Below are some typical examples\n"
    "{examples}"
)

INPUT_TEMPLATE = "Current layout: {input_layout} Input keywords: {input_keywords}"
EXAMPLE_TEMPLATE = "Current layout: {input_layout} Input keywords: {input_keywords} Output: {output_layout}"

LAYOUT_EXAMPLES = [
    {
        "input": {
            "words": ["Hello"],
            "polygons": [[(128, 128), (384, 128), (384, 192), (128, 192)]],
        },
        "output": {
            "words": ["world!"],
            "polygons": [[(128, 192), (384, 192), (384, 256), (128, 256)]],
        },
    },
]
