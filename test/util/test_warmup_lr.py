import sys
import torch.optim as optim
import torch
import unittest

sys.path.insert(0, sys.path[0] + "/../../")

from src.util.warmup_lr import WarmupLR


class TestWarmupLR(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer = optim.SGD([torch.tensor(1.0)], lr=0.1)
        self.scheduler = WarmupLR(
            self.optimizer, initial_lr=0.1, target_lr=1.1, warmup_epochs=5
        )

    def test_lr_values_during_warmup(self):
        for epoch in range(6):
            print(self.optimizer.param_groups[0]["lr"])
            print(self.scheduler.last_epoch)
            self.assertAlmostEqual(
                self.optimizer.param_groups[0]["lr"], 0.1 + epoch * 0.2
            )
            self.scheduler.step()

    def test_lr_values_after_warmup(self):
        [self.scheduler.step() for _ in range(5)]
        for _ in range(5):
            print(self.optimizer.param_groups[0]["lr"])
            print(self.scheduler.last_epoch)
            self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 1.1)
            self.scheduler.step()

    def tearDown(self) -> None:
        self.optimizer = None
        self.scheduler = None


if __name__ == "__main__":
    unittest.main()
