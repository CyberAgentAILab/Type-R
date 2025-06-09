import logging
import os
import re
import time
from typing import Any

import datasets
import hydra
import openai
from hydra.utils import call
from loguru import logger
from omegaconf import DictConfig

logging.getLogger("datasets.fingerprint").setLevel(logging.ERROR)


def simple_adapt_prompt(prompt):
    prompt_adapt = prompt.replace('"', "'")
    matches = re.findall(r"'(.*?)'", prompt_adapt)  # find the keywords enclosed by ''
    words = []
    if matches:
        for match in matches:
            words.extend(match.split())
    if matches:
        for match in matches:
            prompt_adapt = prompt_adapt.replace(f"'{match}'", f"{match}")
    if prompt_adapt[0] == " ":
        prompt_adapt = prompt_adapt[1:]
    if prompt_adapt[-1] == " ":
        prompt_adapt = prompt_adapt[:-1]
    # prompt_adapt = f"Draw a picture about {prompt_adapt} with the large text "
    if len(matches) > 0:
        prompt_adapt = f"Draw a picture about {prompt_adapt} with the large text "
        rendering_text = ""
        for i, match in enumerate(matches):
            rendering_text += f"{match}"
            if i != len(matches) - 1:
                rendering_text += " "
        prompt_adapt += f'"{rendering_text}".'
    else:
        prompt_adapt = f"Draw a picture about {prompt_adapt}."
    return prompt_adapt


def get_augmented_prompt(
    prompt: str,
    client: Any,
    model_name: str,
    max_retry: int,
    retry_interval: float,
):
    messages_base = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]
    blocks_target = get_blocks(prompt)
    messages = messages_base + [
        {
            "role": "user",
            "content": TEMPLATE.format(
                input=prompt,
                quotes=",".join([f"'{t}'" for t in blocks_target]),
            ),
        }
    ]

    n_retry = 0
    diff_word_length = 1000
    simple_adapted_prompt = simple_adapt_prompt(prompt)
    while n_retry < max_retry:
        time.sleep(retry_interval)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            augmented_prompt = response.choices[0].message.content

        except openai.BadRequestError as e:
            print(e)
            n_retry += 1
            continue
        if augmented_prompt is None:
            return simple_adapted_prompt, 1

        # check if all the quotes are mentioned in the augmented prompt
        blocks_pred = get_blocks(augmented_prompt)
        _blocks_pred = []
        for text in blocks_pred:
            _blocks_pred.extend(text.split())
        _blocks_target = []
        for text in blocks_target:
            _blocks_target.extend(text.split())

        def word_filter(word):
            return (
                word.replace(".", "").replace(",", "").replace("'", "").replace('"', "")
            )

        _blocks_pred = list(set(sorted([word_filter(w.lower()) for w in _blocks_pred])))
        _blocks_target = list(set(sorted([w.lower() for w in _blocks_target])))
        _diff_word_length = abs(len(set(_blocks_target) - set(_blocks_pred)))
        if _diff_word_length < diff_word_length:
            diff_word_length = _diff_word_length
            simple_adapted_prompt = augmented_prompt
        if "".join(blocks_target) == "".join(blocks_pred):
            break
        else:
            logger.warning(
                f"Blocks mismatch: {_blocks_pred=} {_blocks_target=} {augmented_prompt=}"
            )
            n_retry += 1

    if n_retry == max_retry:
        logger.error(f"Failed to generate augmented prompt for {prompt=}")
        augmented_prompt = simple_adapted_prompt.replace("\n", "")
        return augmented_prompt

    # make sure a single prompt is in a single line
    augmented_prompt = augmented_prompt.replace("\n", "")
    return augmented_prompt


def get_blocks(text: str) -> list[str]:
    blocks = []
    matches = re.findall(r"'(.*?)'", text)  # find the keywords enclosed by ''
    if matches:
        for match in matches:
            blocks.append(match)
    matches = re.findall(r'"(.*?)"', text)  # find the keywords enclosed by ""
    if matches:
        for match in matches:
            blocks.append(match)
    return list(set(blocks))


TEMPLATE = "'Input': {input}\n'Quotes': {quotes}\n"
SYSTEM_PROMPT = (
    "You are an excellent autonomous AI Assistant."
    "Given a short prompt, generate a concise yet expressive augmented prompt, which will be used as the input of text-to-image models."
    "The augmented prompt should at least follow some rules: "
    "(1) It should include references to data domains such as advertisements, notes, posters, covers, memes, logos, and books."
    "(2) It should mention as many features as possible, such as objects and their composition, colors, and overall atmosphere."
    "(3) All text enclosed in single or double quotes (e.g., 'Michael Jordan') should be displayed legibly in the image, while any other text should not be included. The augmented prompt must specify all text intended for display by using either single or double quotes."
    "(4) Quotation marks are solely for indicating text to be drawn in the image and should not be used for any other purposes, such as possessives."
    "(5) A simpler design organization is preferable."
    "(6) For any other text, interpret the context from the short input and feel free to expand where appropriate."
)


class PromptAugmentor:
    def __init__(self, client, model_name, max_retry, retry_interval):
        self.client = client
        self.model_name = model_name
        self.max_retry = max_retry
        self.retry_interval = retry_interval

    def replace_prompt(self, element):
        prompt = element["prompt"]
        try:
            augmented_prompt = get_augmented_prompt(
                prompt,
                self.client,
                self.model_name,
                self.max_retry,
                self.retry_interval,
            )
        except:
            augmented_prompt = simple_adapt_prompt(prompt)
        element["augmented_prompt"] = augmented_prompt
        return element


@hydra.main(
    version_base=None, config_path="../config", config_name="prompt_augmentation"
)
def augment(config: DictConfig):
    ##########################
    # I/O
    ##########################
    root_res = config.output_dir
    os.makedirs(os.path.join(root_res, "prompt"), exist_ok=True)

    ##########################
    # Load settings
    ##########################
    if config.use_azure:
        client = openai.AzureOpenAI()
        model_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME") or ""
    else:
        client = openai.OpenAI()
        model_name = "gpt-4o"
    prompt_augmentor = PromptAugmentor(
        client, model_name, config.max_retry, config.retry_interval
    )

    ##########################
    # main processings
    ##########################
    if config.input_format == "txt":
        hfds = call(config.dataset)()()
        hfds = hfds.map(prompt_augmentor.replace_prompt)
        with open(
            os.path.join(config.output_dir, "prompt", "augmented_prompts.txt"), "w"
        ) as f:
            for example in hfds:
                f.write(f"{example['augmented_prompt']}\n")
    elif config.input_format == "hfds":
        hfds = call(config.dataset)(subset="ALL")()
        logger.info(f"{hfds=}")
        splits = hfds.keys()
        hfds_new = {}
        for split in splits:
            _hfds = hfds[split].map(prompt_augmentor.replace_prompt)
            hfds_new[split] = _hfds
        hfds_new = datasets.DatasetDict(hfds_new)
        hfds_new.save_to_disk(os.path.join(config.output_dir, "prompt"))
    else:
        raise ValueError(f"Invalid input_format: {config.input_format}")


if __name__ == "__main__":
    logger.info("start prompt augmentation")
    augment()
