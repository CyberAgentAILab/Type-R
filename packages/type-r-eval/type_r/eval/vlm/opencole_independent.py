import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from type_r.eval.vlm.tester import RatingTester
from type_r.eval.vlm.util import validate_path
from type_r.eval.vlm.vlmeval_helper import VLMEvalWrapper

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

BASE_PROMPT = (
    "You are an autonomous AI Assistant who aids designers by providing insightful, objective, and constructive critiques of graphic design projects. "
    "Your goals are: Deliver comprehensive and unbiased evaluations of graphic designs based on established design principles and industry standards. "
    "Identify potential areas for improvement and suggest actionable feedback to enhance the overall aesthetic and effectiveness of the designs. "
    "Maintain a consistent and high standard of critique. "
    "Utilize coordinate information for data description relative to the upper left corner of the image, with the upper left corner serving as the origin, the right as the positive direction, and the downward as the positive direction. "
    "Here are the criteria for evaluating the design:\n"
    "- design_and_layout: The graphic design should present a clean, balanced, and consistent layout. The organization of elements should enhance the message, with clear paths for the eye to follow. A score of 10 signifies a layout that maximizes readability and visual appeal, while a 1 indicates a cluttered, confusing layout with no clear hierarchy or flow.\n"
    "- content_relevance_and_effectiveness: The content should be not only relevant to its purpose but also engaging for the intended audience, effectively communicating the intended message. A score of 10 means the content resonates with the target audience, aligns with the design’s purpose, and enhances the overall message.  A score of 1 indicates the content is irrelevant or does not connect with the audience.\n"
    "- typography_and_color_scheme: Typography and Color Scheme: Typography and color should work together to enhance readability and harmonize with other design elements. This includes font selection, size, line spacing, color, and placement, as well as the overall color scheme of the design.  A score of 10 represents excellent use of typography and color that aligns with the design’s purpose and aesthetic, while a score of 1 indicates poor use of these elements that hinders readability or clashes with the design. If the image does not contain any text, please give 1.\n"
    "- graphics_and_images: Graphics and Images: Any graphics or images used should enhance the design rather than distract from it. They should be high quality, relevant, and harmonious with other elements. A score of 10 indicates graphics or images that enhance the overall design and message, while a 1 indicates low-quality, irrelevant, or distracting visuals.\n"
    "- innovation_and_originality: Innovation and Originality: The design should display an original, creative approach. It should not just follow trends but also show a unique interpretation of the brief. A score of 10 indicates a highly creative and innovative design that stands out in its originality, while a score of 1 indicates a lack of creativity or a generic approach.\n"
    "Please abide by the following rules: "
    "Strive to score as objectively as possible. "
    "Grade seriously. A flawless design can earn 10 points, a mediocre design can only earn 7 points, a design with obvious shortcomings can only earn 4 points, and a very poor design can only earn 1-2 points. "
    "Keep your reasoning concise when rating, and describe it as briefly as possible."
    "If the output is too long, it will be truncated. "
    "Grade each criteria independently. Try not to consider the other criteria when grading a specific one. "
)


CRITERIAS = [
    "design_and_layout",
    "content_relevance_and_effectiveness",
    "typography_and_color_scheme",
    "graphics_and_images",
    "innovation_and_originality",
]

QUESTION = "Rate this image in terms of {criteria}."


@hydra.main(version_base=None, config_path="config", config_name="opencole_independent")
def main(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)()

    first_n = getattr(config, "first_n", None)  # trigger by passing +first_n=<FIRST_N>

    for criteria in CRITERIAS:
        tester = RatingTester(
            system_prompt=BASE_PROMPT,
            question=QUESTION.format(criteria=criteria),
            append_t2i_input=True
            if criteria == "content_relevance_and_effectiveness"
            else False,
        )

        model = VLMEvalWrapper(
            vlm_name="GPT4o",
            output_object=tester.pydantic_object,
            system_prompt=tester.system_prompt,
            # **parse_unknown(unknown),  # TODO: how to feed vlm-specific arguments?
        )
        output_path = validate_path(str(Path(config.output_dir) / f"{criteria}.csv"))

        results: list[dict] = []
        for i, input_ in enumerate(dataset):
            result = tester(model=model, input_=input_)
            results.append(result)
            if first_n is not None and i >= first_n - 1:
                break

        df = pd.DataFrame.from_dict(results)
        df.to_csv(str(output_path), index=False)


if __name__ == "__main__":
    main()
