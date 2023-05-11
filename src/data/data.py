import importlib

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class MyDataset:
    """Class for handling datasets and returning dataloaders."""

    def __init__(
        self, name: str, path: str, transformations: list[dict], load_function: dict
    ):
        self._name = name
        self._path = path

        self._transform = self._extract_transform(transformations)
        self._load_function, self._lf_type = self._extract_load_function(load_function)

    def get_dataloaders(
        self, sampler, batch_size, num_workers, **kwargs
    ) -> dict[str, DataLoader]:
        train, test = self.__get_datasets()
        train_loader = DataLoader(
            train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )
        test_loader = DataLoader(
            test,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )
        return {"train": train_loader, "test": test_loader}

    def __get_datasets(self) -> tuple[Dataset, Dataset]:
        assert callable(self._load_function)
        # NOTE: Adjust if load-function requires it.
        if self._lf_type == "built-in":
            train_dataset = self._load_function(
                self._path, transform=self._transform, train=True
            )
            test_dataset = self._load_function(
                self._path, transform=self._transform, train=False
            )
        elif self._lf_type == "generic":
            train_dataset = self._load_function(
                f"{self._path}/train", transform=self._transform
            )
            test_dataset = self._load_function(
                f"{self._path}/test", transform=self._transform
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

    @property
    def get_transforms(self) -> transforms.Compose:
        """Returns a Compose object of transforms to be applied to the dataset."""
        return self._transform

    @property
    def path_to_dataset(self) -> str:
        """Returns the path to the dataset."""
        return self._path

    @property
    def name(self) -> str:
        """Returns the name of the dataset."""
        return self._name
