import lightning as L
from torch.utils.data import DataLoader

class RequestsDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int,
        collate_fun=None,
        prefetch_factor=None
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.dataset_kwargs = dict(dataset_kwargs)
        self.dataset_kwargs.pop("split", None)
        self.workers = workers
        self.collate_fn = collate_fun
        self.prefetch_factor = prefetch_factor

    def train_dataloader(self, shuffle=True):
        return self._make_dloader("train", shuffle=shuffle)

    def val_dataloader(self, shuffle=False):
        return self._make_dloader("val", shuffle=shuffle)

    def test_dataloader(self, shuffle=False):
        return self._make_dloader("test", shuffle=shuffle)

    def _make_dloader(self, split, shuffle=False):
        dataloader_kwargs = {
            "dataset": self.dataset_cls(**self.dataset_kwargs, split=split),
            "shuffle": shuffle,
            "batch_size": self.batch_size,
            "num_workers": self.workers,
            "collate_fn": self.collate_fn,
        }
        if self.workers > 0 and self.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(**dataloader_kwargs)
