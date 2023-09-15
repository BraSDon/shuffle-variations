import sys
import unittest
from unittest.mock import MagicMock, patch
import torch.distributed as dist

sys.path.insert(0, sys.path[0] + "/../../")

from src.main import setup_distributed_training, free_resources
from src.training.stratified_sampler import StratifiedSampler


class TestStratifiedSampler(unittest.TestCase):
    @patch("src.main.dist.get_world_size", return_value=3)
    @patch("src.main.dist.get_rank", return_value=0)
    def setUp(self, m1, m2) -> None:
        assert dist.get_world_size() == 3
        assert dist.get_rank() == 0

        self.dataset = MagicMock()
        self.dataset.__len__.return_value = 12
        self.dataset.targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.batch_size = 2
        self.seed = 1234
        self.sampler = StratifiedSampler(self.dataset, self.batch_size, self.seed)

    def test_init(self):
        self.assertEqual(self.sampler.dataset, self.dataset)
        self.assertEqual(self.sampler.global_batch_size, 6)
        self.assertEqual(len(self.sampler.indices), 4)
        print(self.sampler.indices)

    def test_len(self):
        self.assertEqual(len(self.sampler), 4)

    @patch("src.main.dist.get_world_size", return_value=3)
    @patch("src.main.dist.get_rank", return_value=0)
    def test_get_indices_not_divisible(self, m1, m2):
        indices = self.setUp_get_indices()
        self.assertEqual(len(indices), 4)

    @patch("src.main.dist.get_world_size", return_value=3)
    @patch("src.main.dist.get_rank", return_value=1)
    def test_get_indices_not_divisible_r1(self, m1, m2):
        indices = self.setUp_get_indices()
        self.assertEqual(len(indices), 4)

    @patch("src.main.dist.get_world_size", return_value=3)
    @patch("src.main.dist.get_rank", return_value=2)
    def test_get_indices_not_divisible_r2(self, m1, m2):
        indices = self.setUp_get_indices()
        self.assertEqual(len(indices), 3)

    def setUp_get_indices(self):
        self.dataset.__len__.return_value = 11
        self.dataset.targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        sampler = StratifiedSampler(self.dataset, batch_size=2, seed=1234)
        indices = sampler._StratifiedSampler__get_indices()
        return indices

    def test_sample(self):
        avail_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        size = 6
        sampled_indices, avail_indices = self.sampler._StratifiedSampler__sample(
            avail_indices, size
        )
        self.assertEqual(len(sampled_indices), size)
        self.assertEqual(len(avail_indices), len(self.dataset.targets) - size)
        self.assertTrue(all(i not in avail_indices for i in sampled_indices))

    def test_sample_all(self):
        avail_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        size = len(avail_indices)
        sampled_indices, avail_indices = self.sampler._StratifiedSampler__sample(
            avail_indices, size
        )
        self.assertEqual(len(sampled_indices), size)
        self.assertEqual(len(avail_indices), 0)
        self.assertTrue(all(i not in avail_indices for i in sampled_indices))

    def test_sample_more_than_available(self):
        avail_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        original_size = len(avail_indices)
        size = len(avail_indices) + 1
        sampled_indices, avail_indices = self.sampler._StratifiedSampler__sample(
            avail_indices, size
        )
        self.assertEqual(len(sampled_indices), original_size)
        self.assertEqual(len(avail_indices), 0)
        self.assertTrue(all(i not in avail_indices for i in sampled_indices))

    def test_sample_empty(self):
        avail_indices = []
        size = 6
        sampled_indices, avail_indices = self.sampler._StratifiedSampler__sample(
            avail_indices, size
        )
        self.assertEqual(len(sampled_indices), 0)
        self.assertEqual(len(avail_indices), 0)


# NOTE: All tests are skipped (return without testing) if distributed training
# fails to initialize. To initialize distributed training, run the following command:
# torchrun --nproc_per_node=4 test/training/test_stratified_sampler.py
class TestStratifiedSamplerDistributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            setup_distributed_training("local", "29500")
        except:
            return

    @classmethod
    def tearDownClass(cls) -> None:
        if not dist.is_initialized():
            return
        free_resources()

    def setUp(self) -> None:
        self.dataset = MagicMock()
        self.dataset.__len__.return_value = 8
        self.dataset.targets = [0, 0, 1, 1, 2, 2, 3, 3]
        self.batch_size = 1
        self.seed = 1234
        self.sampler = StratifiedSampler(self.dataset, self.batch_size, self.seed)

    def test_init(self):
        self.assertEqual(self.sampler.dataset, self.dataset)
        self.assertEqual(self.sampler.global_batch_size, 4)
        self.assertEqual(len(self.sampler.indices), 2)
        print(f"[GPU{dist.get_rank()}]: {self.sampler.indices}")


if __name__ == "__main__":
    unittest.main()
