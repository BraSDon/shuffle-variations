import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torchvision.datasets import CIFAR10

sys.path.insert(0, sys.path[0] + "/../../")

from src.main import setup_distributed_training, free_resources
from src.util.cases import CaseFactory
from src.training.custom_sampler import CustomDistributedSampler
from src.data.sorted_dataset import SortedDataset


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
        free_resources()

    def test_init(self):
        if not dist.is_initialized():
            return
        # Arrange
        dataset = self.train
        case = CaseFactory.create_case("asis_seq_local")
        seed = 1234

        # Act
        sampler = CustomDistributedSampler(dataset, case, seed)

        # Assert
        with patch("src.training.custom_sampler.wandb") as mock_wandb:
            mock_wandb.log = (
                MagicMock()
            )  # Creating a mock of wandb.log() to avoid errors.
            self.assertEqual(sampler.world_size, dist.get_world_size())
            self.assertEqual(sampler.rank, dist.get_rank())
            self.assertEqual(sampler.total_size % sampler.world_size, 0)
            self.assertEqual(len(sampler.indices), sampler.total_size // sampler.world_size)

    def test_iter_shuffle(self):
        if not dist.is_initialized():
            return
        # Arrange
        dataset = self.train
        case = CaseFactory.create_case("asis_seq_local")
        seed = 1234
        sampler = CustomDistributedSampler(dataset, case, seed)
        with patch("src.training.custom_sampler.wandb") as mock_wandb:
            mock_wandb.log = (
                MagicMock()
            )  # Creating a mock of wandb.log() to avoid errors.
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
        case = CaseFactory.create_case("asis_seq_noshuffle")
        seed = 1234
        sampler = CustomDistributedSampler(dataset, case, seed)
        with patch("src.training.custom_sampler.wandb") as mock_wandb:
            mock_wandb.log = (
                MagicMock()
            )  # Creating a mock of wandb.log() to avoid errors.
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
        case = CaseFactory.create_case("asis_seq_local")
        seed = 1234
        sampler = CustomDistributedSampler(dataset, case, seed)

        # Act
        with patch("src.training.custom_sampler.wandb") as mock_wandb:
            mock_wandb.log = (
                MagicMock()
            )  # Creating a mock of wandb.log() to avoid errors.
            seed_tensor = torch.tensor(sampler.seed)
            all_seeds = [
                torch.zeros_like(seed_tensor) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(all_seeds, seed_tensor)
            unique_seeds = set(all_seeds)

            # Assert
            self.assertEqual(len(unique_seeds), dist.get_world_size())

    def test_indices(self):
        if not dist.is_initialized():
            return
        # Arrange
        dataset = SortedDataset(self.train)
        asis_seq_case = CaseFactory.create_case("asis_seq_local")
        asis_step_case = CaseFactory.create_case("asis_step_local")

        # Act
        with patch("src.training.custom_sampler.wandb") as mock_wandb:
            mock_wandb.log = (
                MagicMock()
            )  # Creating a mock of wandb.log() to avoid errors.
            asis_seq_sampler = CustomDistributedSampler(dataset, asis_seq_case, 1234)
            asis_step_sampler = CustomDistributedSampler(dataset, asis_step_case, 1234)
            asis_seq_indices = asis_seq_sampler.indices
            asis_step_indices = asis_step_sampler.indices

            # Assert
            # Rank 0 needs to have the first 25% of the indices
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            total_indices = list(range(len(dataset)))
            self.assertEqual(len(asis_seq_indices), len(dataset) // world_size)
            self.assertEqual(len(asis_step_indices), len(dataset) // world_size)

            self.assertEqual(total_indices[rank::world_size], asis_step_indices)
            self.assertEqual(
                total_indices[
                    rank * len(asis_seq_indices) : (rank + 1) * len(asis_seq_indices)
                ],
                asis_seq_indices,
            )


if __name__ == "__main__":
    unittest.main()
