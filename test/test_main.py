import sys
import unittest
from unittest.mock import patch

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, SequentialLR


sys.path.insert(0, sys.path[0] + "/../")

from src.main import get_scheduler
from src.util.warmup_lr import WarmupLR


class TestMain(unittest.TestCase):
    def __set_up_get_scheduler(self):
        self.optimizer = torch.optim.SGD([torch.tensor(1.0)], lr=0.1)
        self.run_config = {
            "schedulers": {
                "name": "reduce-on-plateau",
                "kwargs": {"mode": "min", "patience": 5},
                "warmup-epochs": 5,
                "reference-kn": 4096,
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
        assert isinstance(scheduler._schedulers[0], WarmupLR)
        assert isinstance(scheduler._schedulers[1], ReduceLROnPlateau)


if __name__ == "__main__":
    unittest.main()
