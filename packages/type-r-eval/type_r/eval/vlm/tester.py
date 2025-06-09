import random
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, create_model

from .util import TesterInput, get_format_instructions
from .vlmeval_helper import VLMEvalWrapper


class BaseTester(BaseModel):
    system_prompt: str
    question: str
    append_t2i_input: bool = False  # Whether to append text input to the model.
    pydantic_object: BaseModel | None = None


class RatingTester(BaseTester):
    def model_post_init(self, ctx: Any) -> None:
        # adaptive pydantic object generation for parsing output
        pydantic_kwargs = {}
        pydantic_kwargs["explanation"] = (str, Field(...))
        pydantic_kwargs["rating"] = (int, Field(ge=1, le=10))

        self.pydantic_object = create_model("RatingOutput", **pydantic_kwargs)
        system_prompt = f"{self.system_prompt} {get_format_instructions(self.pydantic_object.model_json_schema())}"
        self.system_prompt = system_prompt

    def get_mock(self):
        answer = {"explanation": "", "rating": 1}
        return self.pydantic_object.model_validate(answer)

    def __call__(
        self,
        model: VLMEvalWrapper,
        input_: TesterInput,
    ) -> dict[str, Any]:
        assert len(input_.image_paths) == 1, "Rating requires exactly 1 image."

        user_messages = [self.question]
        user_messages.extend(["Image: ", str(input_.image_paths[0])])
        if self.append_t2i_input:
            assert hasattr(input_, "prompt") and input_.prompt is not None
            user_messages.append(f"Text description: {input_.prompt}")

        logger.info(f"{user_messages=}")
        output = model(user_messages)
        if output is None:
            output = self.get_mock()

        result = {
            "id": input_.id,
            "score": output.rating,
            "explanation": output.explanation,
        }
        return result


class VotingTester(BaseTester):
    answer_type: str = (
        "single"  # Voting may pick single or multiple samples as the best.
    )

    def model_post_init(self, ctx: Any) -> None:
        # adaptive pydantic object generation for parsing output
        pydantic_kwargs = {}
        pydantic_kwargs["explanation"] = (str, Field(...))
        # TODO: need validation on range of ids?
        if self.answer_type == "single":
            pydantic_kwargs["choice"] = (int, Field(...))
        elif self.answer_type == "multiple":
            pydantic_kwargs["choices"] = (list[int], Field(...))
        else:
            raise ValueError(f"Invalid answer_type: {self.answer_type}")

        self.pydantic_object = create_model("VotingOutput", **pydantic_kwargs)
        system_prompt = f"{self.system_prompt} {get_format_instructions(self.pydantic_object.model_json_schema())}"
        self.system_prompt = system_prompt

    def get_mock(self, N: int):
        if self.answer_type == "single":
            answer = {"explanation": "", "choice": random.choice(list(range(N)))}
        elif self.answer_type == "multiple":
            answer = {"explanation": "", "choices": list(range(N))}
        else:
            raise ValueError(f"Invalid answer_type: {self.answer_type}")
        return self.pydantic_object.model_validate(answer)

    def __call__(
        self,
        model: VLMEvalWrapper,
        input_: TesterInput,
    ) -> dict[str, Any]:
        N = len(input_.image_paths)
        assert isinstance(input_.image_paths, list) and all(
            [isinstance(t, str) for t in input_.image_paths]
        )
        assert N >= 2, "Voting requires at least 2 images."
        indexes = random.sample(range(N), N)
        user_messages = [self.question]

        for i in range(N):
            user_messages.extend(
                [f" image id: {i + 1}", str(input_.image_paths[indexes[i]])]
            )

        if self.append_t2i_input:
            assert hasattr(input_, "prompt")
            user_messages.append(f"Text description: {input_.prompt}")

        logger.info(f"{user_messages=}")
        output = model(user_messages)
        if output is None:
            output = self.get_mock(N)

        if self.answer_type == "single":
            choices = set([output.choice - 1])
        elif self.answer_type == "multiple":
            choices = set([o - 1 for o in output.choices])

        result = {}
        result["id"] = Path(input_.image_paths[0]).stem
        names = []
        for i in range(N):
            name = str(Path(input_.image_paths[indexes[i]]).parent)
            # if the model thinks all the images are tied, we will assign 1 to all
            if len(choices) == 0:
                result[f"result:{name}"] = 1
            else:
                if i in choices:
                    result[f"result:{name}"] = 1
                else:
                    result[f"result:{name}"] = 0
            names.append(name)
        for i, name in enumerate(names):
            result[f"explanation:image {i + 1}"] = name
        result["explanation"] = output.explanation

        return result


class OpenCOLERatingTester(BaseTester):
    def model_post_init(self, ctx: Any) -> None:
        # adaptive pydantic object generation for parsing output
        pydantic_kwargs = {}
        pydantic_kwargs["design_and_layout"] = (int, Field(ge=1, le=10))
        pydantic_kwargs["content_relevance_and_effectiveness"] = (
            int,
            Field(ge=1, le=10),
        )
        pydantic_kwargs["typography_and_color_scheme"] = (int, Field(ge=1, le=10))
        pydantic_kwargs["graphics_and_images"] = (int, Field(ge=1, le=10))
        pydantic_kwargs["innovation_and_originality"] = (int, Field(ge=1, le=10))

        self.pydantic_object = create_model("OpenCOLERateOutput", **pydantic_kwargs)
        system_prompt = f"{self.system_prompt}"
        self.system_prompt = system_prompt

    def get_mock(self):
        answer = {
            "design_and_layout": 1,
            "content_relevance_and_effectiveness": 1,
            "typography_and_color_scheme": 1,
            "graphics_and_images": 1,
            "innovation_and_originality": 1,
        }
        return self.pydantic_object.model_validate(answer)

    def __call__(
        self,
        model: VLMEvalWrapper,
        input_: TesterInput,
    ) -> dict[str, Any]:
        assert len(input_.image_paths) == 1, "Rating requires exactly 1 image."

        user_messages = [self.question]
        user_messages.extend(["Image: ", str(input_.image_paths[0])])
        if self.append_t2i_input:
            assert hasattr(input_, "prompt")
            user_messages.append(f"Text description: {input_.prompt}")

        logger.info(f"{user_messages=}")
        output = model(user_messages)
        if output is None:
            output = self.get_mock()

        result = {
            "id": input_.id,
            "design_and_layout": output.design_and_layout,
            "content_relevance_and_effectiveness": output.content_relevance_and_effectiveness,
            "typography_and_color_scheme": output.typography_and_color_scheme,
            "graphics_and_images": output.graphics_and_images,
            "innovation_and_originality": output.innovation_and_originality,
        }
        return result
