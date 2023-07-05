import sys
import unittest
import torch.distributed as dist
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, sys.path[0] + "/../../")
from src.data.data import MyDataset
from src.main import setup_distributed_training

# NOTE: All tests are skipped (return without testing) if distributed training
# fails to initialize. To initialize distributed training, run the following command:
# torchrun --nproc_per_node=4 test/data/test_data.py
class TestMyDataset(unittest.TestCase):
    data_path = "../../data"

    @classmethod
    def setUpClass(cls) -> None:
        # Ensure that the CIFAR10 dataset is downloaded.
        CIFAR10(root=cls.data_path, train=True, download=True)
        CIFAR10(root=cls.data_path, train=False, download=True)

        try:
            setup_distributed_training("local", "29500")
        except:
            return
        dist.barrier()

    def test_get_dataloaders(self):
        train_transformations = [
            {"name": "ToTensor", "kwargs": {}},
        ]
        with patch("src.data.data.wandb") as mock_wandb:
            mock_wandb.log = (
                MagicMock()
            )  # Creating a mock of wandb.log() to avoid errors.
            dataset = MyDataset("CIFAR10", self.data_path, train_transformations, [], {
                "module": "torchvision.datasets.cifar",
                "type": "built-in",
                "name": "CIFAR10",
            }, 10, torch.device("cpu"))
        batch_size = 32
        sampler = MagicMock()
        num_workers = 4

        train_loader = dataset.get_train_loader(sampler, batch_size, num_workers)
        test_loader = dataset.get_test_loader(sampler, batch_size, num_workers)

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        self.assertEqual(len(train_loader.dataset), 50000)
        self.assertEqual(len(test_loader.dataset), 10000)

        self.assertEqual(train_loader.batch_size, batch_size)
        self.assertEqual(test_loader.batch_size, batch_size)

        # Iterate over the dataloaders to check if they are working.
        for _ in train_loader:
            break
        for _ in test_loader:
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
