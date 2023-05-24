import importlib
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import transforms

from src.data.sorted_dataset import SortedDataset


class MyDataset:
    """Class for handling datasets and returning dataloaders."""

    def __init__(self, name: str, path: str, train_transformations: list[dict], test_transformations: list[dict],
                 load_function: dict, num_classes: int):
        self.name = name
        self.path = path
        self.num_classes = num_classes

        self.train_transform = self._extract_transform(train_transformations)
        self.test_transform = self._extract_transform(test_transformations)
        self.load_function, self._lf_type = self._extract_load_function(load_function)

        self.train_dataset, self.test_dataset = self.__get_datasets()
        self.sort_train_dataset()  # Make sure that train_dataset is sorted by class.
        self.train_label_frequencies = self._get_train_label_frequencies()

    def get_train_loader(
        self, sampler: Sampler, batch_size: int, num_workers: int, **kwargs
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )

    def get_test_loader(
        self, sampler: Sampler, batch_size: int, num_workers: int, **kwargs
    ) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )

    def __get_datasets(self) -> tuple[Dataset, Dataset]:
        assert callable(self.load_function)
        # NOTE: Adjust if load-function requires it.
        if self._lf_type == "built-in":
            train_dataset = self.load_function(
                self.path, transform=self.train_transform, train=True
            )
            test_dataset = self.load_function(
                self.path, transform=self.test_transform, train=False
            )
        elif self._lf_type == "generic":
            train_dataset = self.load_function(
                f"{self.path}/train", transform=self.train_transform
            )
            test_dataset = self.load_function(
                f"{self.path}/test", transform=self.test_transform
            )
        else:
            raise ValueError(f"Unknown load function type: {self._lf_type}")
        return train_dataset, test_dataset

    def sort_train_dataset(self):
        self.train_dataset = SortedDataset(self.train_dataset)

    def _get_train_label_frequencies(self):
        # Calculate label frequency for train dataset.
        total_freq = torch.zeros(self.num_classes)
        dl = DataLoader(self.train_dataset, batch_size=32, num_workers=4)
        for _, labels in dl:
            freq = torch.bincount(labels, minlength=self.num_classes)
            total_freq += freq
        return total_freq / total_freq.sum()

    @staticmethod
    def _extract_transform(transformations: list[dict]) -> transforms.Compose:
        transform_list = []
        for t in transformations:
            transform_class = getattr(transforms, t["name"])
            kwargs = t["kwargs"] if "kwargs" in t else {}
            transform_list.append(transform_class(**kwargs))
        return transforms.Compose(transform_list)

    @staticmethod
    def _extract_load_function(load_function: dict) -> tuple[callable, str]:
        module, lf_type, name = (
            load_function["module"],
            load_function["type"],
            load_function["name"],
        )

        # Import specified module and
        # get the function in that module with the specified name.
        module = importlib.import_module(module)
        load_function = getattr(module, name)
        return load_function, lf_type
