import datasets
from datasets import Dataset

from .base import BaseDataset


class PromptHFDS(BaseDataset):
    def __init__(
        self,
        prompt_file: str,
        *args,
        **kwargs,
    ) -> None:
        with open(
            prompt_file,
            "r",
        ) as fr:
            prompts = fr.readlines()
            prompts = [_.strip() for _ in prompts]
        self.prompts = prompts

    def load_hfds(self) -> datasets.Dataset:
        data_list = []
        for idx, prompt in enumerate(self.prompts):
            data_list.append({"dataset_name": "demo", "id": idx, "prompt": prompt})

        hfds = Dataset.from_list(data_list)
        return hfds

    def dataset_filter(self, hfds: datasets.Dataset) -> datasets.Dataset:
        return hfds
