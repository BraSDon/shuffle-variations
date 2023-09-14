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

    # TODO: Test edge cases. Specifically if the number of samples is not
    #  divisible by global_batch_size!


if __name__ == "__main__":
    unittest.main()
