import unittest
from unittest.mock import MagicMock
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.data.data import MyDataset


class TestMyDataset(unittest.TestCase):
    data_path = "../../data"

    @classmethod
    def setUpClass(cls) -> None:
        # Ensure that the CIFAR10 dataset is downloaded.
        CIFAR10(root=cls.data_path, train=True, download=True)
        CIFAR10(root=cls.data_path, train=False, download=True)

    def test_get_dataloaders(self):
        dataset = MyDataset(
            "CIFAR10",
            self.data_path,
            [],
            {
                "module": "torchvision.datasets.cifar",
                "type": "built-in",
                "name": "CIFAR10",
            },
        )
        batch_size = 32
        sampler = MagicMock()
        num_workers = 4
        dataloaders = dataset.get_dataloaders(sampler, batch_size, num_workers)

        self.assertIsInstance(dataloaders["train"], DataLoader)
        self.assertIsInstance(dataloaders["test"], DataLoader)

        self.assertEqual(len(dataloaders["train"].dataset), 50000)
        self.assertEqual(len(dataloaders["test"].dataset), 10000)

        self.assertEqual(dataloaders["train"].batch_size, batch_size)
        self.assertEqual(dataloaders["test"].batch_size, batch_size)

        # Iterate over the dataloaders to check if they are working.
        for _ in dataloaders["train"]:
            break
        for _ in dataloaders["test"]:
            break

    def test__extract_transform(self):
        transformations = [
            {"name": "RandomCrop", "kwargs": {"size": (32, 32), "padding": 4}},
            {"name": "RandomHorizontalFlip"},
            {"name": "ToTensor"},
        ]
        transform = MyDataset._extract_transform(transformations)
        self.assertIsInstance(transform, transforms.Compose)
        self.assertEqual(len(transform.transforms), 3)
        self.assertIsInstance(transform.transforms[0], transforms.RandomCrop)
        self.assertIsInstance(transform.transforms[1], transforms.RandomHorizontalFlip)
        self.assertIsInstance(transform.transforms[2], transforms.ToTensor)
        self.assertEqual(transform.transforms[0].size, (32, 32))

    def test__extract_load_function(self):
        load_function = {
            "module": "torchvision.datasets.cifar",
            "type": "built-in",
            "name": "CIFAR10",
        }
        lf, lf_type = MyDataset._extract_load_function(load_function)
        self.assertEqual(lf_type, "built-in")
        self.assertEqual(lf, CIFAR10)


if __name__ == "__main__":
    unittest.main()