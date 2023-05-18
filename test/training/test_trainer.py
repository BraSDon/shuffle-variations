import sys
import unittest
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, sys.path[0] + "/../../")
from src.main import setup_distributed_training
from src.training.train import Trainer


# NOTE: All tests are skipped (return without testing) if distributed training
# fails to initialize. To initialize distributed training, run the following command:
# torchrun --nproc_per_node=4 test/training/test_trainer.py
class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            setup_distributed_training("local", "29500")
        except:
            return
        dist.barrier()

    def setUp(self):
        if not dist.is_initialized():
            return
        self.model = DummyModel()
        self.optimizer = MagicMock()
        self.criterion = MagicMock()
        self.train_loader = MagicMock()
        self.test_loader = MagicMock()
        self.system = "local"

        # Create the Trainer instance
        self.trainer = Trainer(
            self.model,
            self.optimizer,
            self.criterion,
            self.train_loader,
            self.test_loader,
            self.system,
        )

    def test_init(self):
        if not dist.is_initialized():
            return

        if torch.cuda.is_available():
            self.assertEqual(self.trainer.device, torch.device("cuda"))
        else:
            self.assertEqual(self.trainer.device, torch.device("cpu"))

    def test_average_statistic(self):
        if not dist.is_initialized():
            return
        # Arrange
        statistic = dist.get_rank()

        # Act
        average_statistic = self.trainer.average_statistic(statistic)

        # Assert
        self.assertEqual(average_statistic, (0 + 1 + 2 + 3) / 4.0)

    def test_calculate_accuracy(self):
        if not dist.is_initialized():
            return
        # Outputs is a tensor of shape (batch_size, num_classes): (2, 10)
        # Labels is a tensor of shape (batch_size): (2)
        outputs = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0],
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            ]
        )
        labels_top1_100 = torch.tensor([8, 0])
        labels_top1_50 = torch.tensor([8, 1])
        labels_top5_100 = torch.tensor([4, 4])
        labels_top5_50 = torch.tensor([4, 5])

        self.assertEqual(
            self.trainer.calculate_accuracy(outputs, labels_top1_100), [100.0, 100.0]
        )
        self.assertEqual(
            self.trainer.calculate_accuracy(outputs, labels_top1_50), [50.0, 100.0]
        )
        self.assertEqual(
            self.trainer.calculate_accuracy(outputs, labels_top5_100), [0.0, 100.0]
        )
        self.assertEqual(
            self.trainer.calculate_accuracy(outputs, labels_top5_50), [0.0, 50.0]
        )


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    unittest.main()
