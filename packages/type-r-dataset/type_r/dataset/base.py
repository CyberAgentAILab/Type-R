import datasets


class BaseDataset:
    def __init__(*args, **kwargs) -> None:
        pass

    def load_hfds(self) -> datasets.Dataset:
        return NotImplementedError

    def dataset_filter(self, hfds: datasets.Dataset) -> datasets.Dataset:
        raise NotImplementedError

    def __call__(
        self,
    ) -> datasets.Dataset:
        hfds = self.load_hfds()
        return self.dataset_filter(hfds)
