from pathlib import Path
from typing import Any

from .util import TesterInput


class DirectoryDataset:
    def __init__(
        self,
        image_dir: str
        | list[str],  # image_dir=tmp/dirA or image_dir=[tmp/dirA,tmp/dirB]
        image_ext: str = "png",
        prompt_dir: str | None = None,
        hfds: str | None = None,
    ) -> None:
        super().__init__()
        assert prompt_dir is None or hfds is None, (
            "Specify either prompt_dir or hfds"
        )  # not both
        self.data: list[dict[str, Any]] = []

        if isinstance(image_dir, str):
            image_dir = [image_dir]

        def get_ids(image_dir):
            ids = None
            for dir_ in image_dir:
                ids_tmp = []
                for f in sorted(list(Path(dir_).glob(f"*.{image_ext}"))):
                    id_ = f.stem
                    ids_tmp.append(id_)

                if ids is None:
                    ids = ids_tmp
                else:
                    assert ids == ids_tmp, "IDs are not common across directories."
            return ids

        if prompt_dir is not None:
            ids = get_ids(image_dir)
            # prompt_files = list(Path(prompt_dir).glob("*.txt"))
            # assert len(prompt_files) == len(
            #     ids
            # ), "IDs are not common across directories."
            for id_ in ids:
                with (Path(prompt_dir) / f"{id_}.txt").open("r") as f:
                    prompt = f.read().strip()

                image_paths = []
                for dir_ in image_dir:
                    image_path = Path(dir_) / f"{id_}.{image_ext}"
                    assert image_path.exists()
                    image_paths.append(str(image_path))
                self.data.append(
                    TesterInput(image_paths=image_paths, prompt=prompt, id=id_)
                )
        elif hfds is not None:
            for x in hfds:
                image_paths = []
                image_id = f"{x['dataset_name']}_{x['id']}"
                for dir_ in image_dir:
                    image_path = Path(dir_) / f"{image_id}.{image_ext}"
                    assert image_path.exists()
                    image_paths.append(str(image_path))
                self.data.append(
                    TesterInput(
                        image_paths=image_paths, prompt=x["prompt"], id=image_id
                    )
                )
        else:
            ids = get_ids(image_dir)
            for id_ in ids:
                image_paths = []
                for dir_ in image_dir:
                    image_path = Path(dir_) / f"{id_}.{image_ext}"
                    assert image_path.exists()
                    image_paths.append(str(image_path))
                self.data.append(TesterInput(image_paths=image_paths, id=id_))

    def __len__(self) -> int:
        n = len(self.data)
        assert n > 0, "Dataset is empty."
        return n

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int) -> TesterInput:
        return self.data[idx]
