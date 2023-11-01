import unittest
import sys
import os

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, sys.path[0] + "/../../")

from src.data.datasets import SUSYDataset

# If SUSY.csv not in ./data, set DATA_NOT_PRESENT to True
DATASET_NOT_PRESENT = False
if not os.path.isfile("./data/SUSY.csv"):
    DATASET_NOT_PRESENT = True


@unittest.skipIf(DATASET_NOT_PRESENT, "SUSY dataset could not be found in ./data")
class TestSUSYDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_root = "./data"
        cls.train_set = SUSYDataset(cls.dataset_root, train=True)
        cls.test_set = SUSYDataset(cls.dataset_root, train=False)
        cls.dataset_size = 5000000

    def test_dataset_length(self):
        self.assertEqual(len(self.train_set), 0.1 * self.dataset_size)

    def test_dataset_item(self):
        data, target = self.train_set[0]
        self.assertEqual(len(data), 18)  # Number of features
        self.assertTrue(isinstance(data, torch.Tensor))
        self.assertTrue(isinstance(target, torch.Tensor))

    def test_data_loader(self):
        dataloader = DataLoader(self.train_set, batch_size=32, shuffle=True)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch), 2)  # Tuple of data and target
        self.assertEqual(
            batch[0].shape, torch.Size([32, 18])
        )  # Batch size x Number of features
        self.assertEqual(batch[1].shape, torch.Size([32]))  # Batch size

    def test_stratified_split(self):
        train_targets = self.train_set.targets
        unique_values, train_class_counts = torch.unique(
            train_targets, return_counts=True
        )

        test_targets = self.test_set.targets
        unique_values, test_class_counts = torch.unique(
            test_targets, return_counts=True
        )

        self.assertEqual(len(train_targets), 0.1 * self.dataset_size)
        self.assertEqual(len(test_targets), 0.9 * self.dataset_size)

        # Check if the class distribution is preserved in the train and test sets
        for class_label in range(2):
            self.assertAlmostEqual(
                train_class_counts[class_label] / len(train_targets),
                test_class_counts[class_label] / len(test_targets),
                delta=0.05,
            )  # Allowing a 5% deviation


if __name__ == "__main__":
    unittest.main()
