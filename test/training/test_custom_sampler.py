import sys
import unittest

import torch
import torch.distributed as dist
from torchvision.datasets import CIFAR10


sys.path.insert(0, sys.path[0] + "/../../")

from src.main import setup_distributed_training, destroy_distributed_training
from src.data.partition import SequentialPartitioner
from src.util.cases import Case
from src.training.custom_sampler import CustomDistributedSampler


# NOTE: All tests are skipped (return without testing) if distributed training
# fails to initialize. To initialize distributed training, run the following command:
# torchrun --nproc_per_node=4 test/training/test_custom_sampler.py
class TestCustomDistributedSampler(unittest.TestCase):
    data_path = "../../data"

    @classmethod
    def setUpClass(cls) -> None:
        try:
            setup_distributed_training("local", "29500")
        except:
            return
        cls.train = CIFAR10(root=cls.data_path, train=True, download=True)
        cls.test = CIFAR10(root=cls.data_path, train=False, download=True)
        dist.barrier()

    @classmethod
    def tearDownClass(cls) -> None:
        if not dist.is_initialized():
            return
        destroy_distributed_training()

    def test_init(self):
        if not dist.is_initialized():
            return
        # Arrange
        dataset = self.train
        case = Case("asis_seq_local", False, SequentialPartitioner(), True)
        seed = 1234

        # Act
        sampler = CustomDistributedSampler(dataset, case, seed)

        # Assert
        self.assertEqual(sampler.world_size, dist.get_world_size())
        self.assertEqual(sampler.rank, dist.get_rank())
        self.assertEqual(sampler.total_size % sampler.world_size, 0)
        self.assertEqual(len(sampler.indices), sampler.total_size // sampler.world_size)

    def test_iter_shuffle(self):
        if not dist.is_initialized():
            return
        # Arrange
        dataset = self.train
        case = Case("asis_seq_local", True, SequentialPartitioner(), True)
        seed = 1234
        sampler = CustomDistributedSampler(dataset, case, seed)
        indices = list(sampler)

        # Act
        indices2 = list(sampler)

        # Assert
        self.assertNotEqual(indices, indices2)

    def test_iter_no_shuffle(self):
        if not dist.is_initialized():
            return
        # Arrange
        dataset = self.train
        case = Case("asis_seq_noshuffle", False, SequentialPartitioner(), False)
        seed = 1234
        sampler = CustomDistributedSampler(dataset, case, seed)
        indices = list(sampler)

        # Act
        indices2 = list(sampler)

        # Assert
        self.assertEqual(indices, indices2)

    def test_seeds_different(self):
        if not dist.is_initialized():
            return
        # Arrange
        dataset = self.train
        case = Case("asis_seq_local", True, SequentialPartitioner(), True)
        seed = 1234
        sampler = CustomDistributedSampler(dataset, case, seed)

        # Act
        seed_tensor = torch.tensor(sampler.seed)
        all_seeds = [
            torch.zeros_like(seed_tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(all_seeds, seed_tensor)
        unique_seeds = set(all_seeds)

        # Assert
        self.assertEqual(len(unique_seeds), dist.get_world_size())


if __name__ == "__main__":
    unittest.main()
