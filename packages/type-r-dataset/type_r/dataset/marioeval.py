import datasets

from .base import BaseDataset


class MarioEvalDataset(BaseDataset):
    def __init__(
        self,
        hfds_dir: str,
        sub_set: str,
        sub_datasetname: str,
        start_index: int,
        end_index: int,
        prompt_file: str | None = None,
        indexlist_file: str | None = None,
        use_augmented_prompt: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.hfds_dir = hfds_dir
        self.sub_set = sub_set
        self.sub_datasetname = sub_datasetname
        self.start_index = start_index
        self.end_index = end_index
        self.prompt_file = prompt_file
        self.use_augmented_prompt = use_augmented_prompt
        self.indexlist_file = indexlist_file

    def load_hfds(self) -> datasets.Dataset:
        if self.sub_set == "ALL":
            return datasets.load_from_disk(self.hfds_dir)
        else:
            return datasets.load_from_disk(self.hfds_dir)[self.sub_set]

    def dataset_filter(self, hfds: datasets.Dataset) -> datasets.Dataset:
        if self.sub_datasetname != "all":
            hfds = hfds.filter(
                lambda example: example["dataset_name"] == self.sub_datasetname
            )
        if self.start_index != -1 and self.end_index != -1:
            hfds = hfds.select(range(self.start_index, self.end_index))
        if self.indexlist_file is not None:
            with open(self.indexlist_file, "r") as f:
                indexlist = f.read().splitlines()
            indexlist = [int(i) for i in indexlist]
            hfds = hfds.select(indexlist)

        if self.prompt_file is not None:

            def replace_prompt(example):
                example["prompt"] = prompts[example["id"]]
                return example

            global prompts
            with open(self.prompt_file, "r") as f:
                prompts = f.read().splitlines()
            hfds = hfds.map(replace_prompt)
        if self.use_augmented_prompt:

            def replace_prompt2augmentedprompt(example):
                example["prompt"] = example["augmented_prompt"]
                return example

            hfds = hfds.map(replace_prompt2augmentedprompt)
        return hfds
