import sys
import unittest
from unittest.mock import MagicMock, patch

import torchmetrics
import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, sys.path[0] + "/../../")
from src.main import setup_distributed_training, get_scheduler
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
        self.my_dataset = MagicMock()
        self.my_dataset.num_classes = 10

        self.run_config = {
            "optimizer": {
                "name": "sgd",
                "kwargs": {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.0001},
            },
            "schedulers": {
                "name": "step",
                "kwargs": {"step_size": 5, "gamma": 0.1},
                "warmup-epochs": 5,
                "reference-kn": 4096,
            },
            "batch-size": 128,
        }

        # Create the Trainer instance
        self.trainer = Trainer(
            self.model,
            self.optimizer,
            self.criterion,
            None,
            self.train_loader,
            self.test_loader,
            self.system,
            self.my_dataset,
            torch.device("cpu"),
        )

    @patch("src.training.train.Trainer.run_epoch")
    @patch("src.training.train.Trainer.test")
    def test_lr_scheduler(self, m1, m2):
        """Test that the lr scheduler is called if it is not None"""
        if not dist.is_initialized():
            return

        self.trainer.scheduler = MagicMock()
        self.trainer.train(1)
        self.trainer.scheduler.step.assert_called_once()

    @patch("src.training.train.Trainer.run_epoch")
    @patch("src.training.train.Trainer.test")
    @patch("src.main.dist.get_world_size", return_value=64)
    def test_lr_scheduler_transition(self, m1, m2, m3):
        """Test that the lr scheduler is called if it is not None"""
        if not dist.is_initialized():
            return

        assert dist.get_world_size() == 64

        self.trainer.optimizer = optim = torch.optim.SGD([torch.tensor(1.0)], lr=0.2)
        self.trainer.scheduler = get_scheduler(optim, self.run_config)
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.1)
        self.trainer.optimizer.step()
        self.trainer.train(4)
        print(f"After 4 epochs: {optim.param_groups[0]['lr']}")
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.18)
        self.trainer.train(1)
        print(f"After 5 epochs: {optim.param_groups[0]['lr']}")
        print(optim.param_groups[0]["lr"])
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.2)
        self.trainer.train(5)
        print(f"After 10 epochs: {optim.param_groups[0]['lr']}")
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.02)

    @patch("src.training.train.Trainer.run_epoch")
    @patch("src.training.train.Trainer.test")
    @patch("src.main.dist.get_world_size", return_value=128)
    def test_lr_scheduler_transition_2(self, m1, m2, m3):
        """Test that the lr scheduler is called if it is not None"""
        if not dist.is_initialized():
            return

        assert dist.get_world_size() == 128

        self.trainer.optimizer = optim = torch.optim.SGD([torch.tensor(1.0)], lr=0.4)
        self.trainer.scheduler = get_scheduler(optim, self.run_config)
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.1)
        self.trainer.optimizer.step()
        self.trainer.train(4)
        print(f"After 4 epochs: {optim.param_groups[0]['lr']}")
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.34)
        self.trainer.train(1)
        print(f"After 5 epochs: {optim.param_groups[0]['lr']}")
        print(optim.param_groups[0]["lr"])
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.4)
        self.trainer.train(5)
        print(f"After 10 epochs: {optim.param_groups[0]['lr']}")
        self.assertAlmostEqual(optim.param_groups[0]["lr"], 0.04)

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

        num_classes = 10
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top1_100, "multiclass", num_classes=num_classes
            ).item(),
            1.0,
        )
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top1_100, "multiclass", top_k=5, num_classes=num_classes
            ).item(),
            1.0,
        )
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top1_50, "multiclass", num_classes=num_classes
            ).item(),
            0.5,
        )
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top1_50, "multiclass", top_k=5, num_classes=num_classes
            ).item(),
            1.0,
        )
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top5_100, "multiclass", num_classes=num_classes
            ).item(),
            0.0,
        )
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top5_100, "multiclass", top_k=5, num_classes=num_classes
            ).item(),
            1.0,
        )
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top5_50, "multiclass", num_classes=num_classes
            ).item(),
            0.0,
        )
        self.assertEqual(
            torchmetrics.functional.accuracy(
                outputs, labels_top5_50, "multiclass", top_k=5, num_classes=num_classes
            ).item(),
            0.5,
        )


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    unittest.main()
