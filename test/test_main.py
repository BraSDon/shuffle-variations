import sys
import unittest
from unittest.mock import patch

import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR

sys.path.insert(0, sys.path[0] + "/../")

from src.main import get_scheduler, get_optimizer


class TestMain(unittest.TestCase):
    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_optimizer(self, m1):
        run_config = {
            "batch-size": 128,
            "optimizer": {
                "name": "rmsprop",
                "kwargs": {
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                    "alpha": 0.99,
                    "eps": 1e-08,
                },
            },
            "schedulers": {"reference-kn": 4096},
        }
        model = torch.nn.Linear(1, 1)
        optimizer = get_optimizer(model, run_config)
        self.assertEqual(optimizer.__class__.__name__, "RMSprop")
        # Check if the optimizer is correctly initialized
        self.assertEqual(optimizer.defaults["lr"], 1.0 / 80.0)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0001)
        self.assertEqual(optimizer.defaults["alpha"], 0.99)
        self.assertEqual(optimizer.defaults["eps"], 1e-08)

    def __set_up_get_scheduler(self):
        self.optimizer = torch.optim.SGD([torch.tensor(1.0)], lr=0.1)
        self.run_config = {
            "schedulers": {
                "name": "step",
                "warmup-epochs": 5,
                "reference-kn": 4096,
                "kwargs": {"gamma": 0.1, "step_size": 30},
            },
            "learning-rate": 0.1,
            "batch-size": 128,
        }

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scheduler(self, mock_dist):
        self.__set_up_get_scheduler()
        scheduler = get_scheduler(self.optimizer, self.run_config)
        assert isinstance(scheduler, SequentialLR)
        assert len(scheduler._schedulers) == 2
        assert isinstance(scheduler._schedulers[0], LinearLR)
        assert isinstance(scheduler._schedulers[1], StepLR)


if __name__ == "__main__":
    unittest.main()
