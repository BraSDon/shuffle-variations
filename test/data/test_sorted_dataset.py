import sys
import unittest
from unittest.mock import MagicMock

import torch

sys.path.insert(0, sys.path[0] + "/../../")

from src.data.sorted_dataset import sort_indices, SortedDataset


class TestSortedDataset(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset with targets for testing
        self.dataset = MagicMock()
        self.dataset.targets = [1, 0, 2, 1, 0, 2, 1, 0, 2, 1]

        # Specify __getitem__ and __len__ for the mock dataset
        self.dataset.__getitem__.side_effect = lambda x: x
        self.dataset.__len__.side_effect = lambda: len(self.dataset.targets)

    def test_sort_indices(self):
        sorted_indices = sort_indices(self.dataset)
        self.assertIsInstance(sorted_indices, torch.Tensor)
        self.assertEqual(sorted_indices.tolist(), [1, 4, 7, 0, 3, 6, 9, 2, 5, 8])

    def test_sorted_dataset(self):
        sorted_dataset = SortedDataset(self.dataset)
        self.assertEqual(sorted_dataset[0], 1)
        self.assertEqual(sorted_dataset[1], 4)
        self.assertEqual(sorted_dataset[2], 7)
        self.assertEqual(sorted_dataset[3], 0)
        self.assertEqual(sorted_dataset[4], 3)
        self.assertEqual(sorted_dataset[5], 6)
        self.assertEqual(sorted_dataset[6], 9)
        self.assertEqual(sorted_dataset[7], 2)
        self.assertEqual(sorted_dataset[8], 5)
        self.assertEqual(sorted_dataset[9], 8)
        self.assertEqual(len(sorted_dataset), 10)


if __name__ == "__main__":
    unittest.main()
