import importlib

from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import transforms


class MyDataset:
    """Class for handling datasets and returning dataloaders."""

    def __init__(
        self, name: str, path: str, transformations: list[dict], load_function: dict
    ):
        self.name = name
        self.path = path

        self.transform = self._extract_transform(transformations)
        self.load_function, self._lf_type = self._extract_load_function(load_function)

        self.train_dataset, self.test_dataset = self.__get_datasets()

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
                self.path, transform=self.transform, train=True
            )
            test_dataset = self.load_function(
                self.path, transform=self.transform, train=False
            )
        elif self._lf_type == "generic":
            train_dataset = self.load_function(
                f"{self.path}/train", transform=self.transform
            )
            test_dataset = self.load_function(
                f"{self.path}/test", transform=self.transform
            )
        else:
            raise ValueError(f"Unknown load function type: {self._lf_type}")
        return train_dataset, test_dataset

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
