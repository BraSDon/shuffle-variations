import argparse
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

import torch
import torch.distributed as dist
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR

sys.path.insert(0, sys.path[0] + "/../")

from src.main import (
    get_scheduler,
    get_optimizer,
    parse_configs,
    get_samplers,
    setup_distributed_training,
    get_scaled_lr,
)


class TestParseConfigs(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = "test-config.yaml"
        self.system_config = {"system": "local"}
        self.run_config = {"case": "pre-step-local"}

        # Create and write to test-config.yaml
        with open(self.config_path, "w") as f:
            f.write("case: pre-step-local")

        # Create and write system-config.yaml
        with open("system-config.yaml", "w") as f:
            f.write("system: local")

    def tearDown(self) -> None:
        time.sleep(2)
        # Remove test-config.yaml
        with open(self.config_path, "w") as f:
            f.write("")
        # Remove system-config.yaml
        with open("system-config.yaml", "w") as f:
            f.write("")

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(config_path="test-config.yaml"),
    )
    def test_parse_configs(self, mock_args):
        expected_run_config = self.run_config
        expected_system_config = self.system_config
        system_config, run_config = parse_configs()
        self.assertEqual(system_config, expected_system_config)
        self.assertEqual(run_config, expected_run_config)

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            config_path="test-config.yaml", case="baseline"
        ),
    )
    def test_parse_configs_edge(self, mock_args):
        with open(self.config_path, "w") as f:
            f.write("batch-size: 128")
        expected_system_config = self.system_config
        system_config, run_config = parse_configs()
        self.assertEqual(system_config, expected_system_config)
        self.assertEqual(run_config, {"batch-size": 128, "case": "baseline"})


# torchrun --nproc_per_node=4 test/test_main.py
class TestGetSamplers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            setup_distributed_training("local", "29500")
        except:
            return
        dist.barrier()

    def setUp(self) -> None:
        self.dataset = MagicMock()
        self.seed = 1

    def test_get_samplers_baseline(self):
        if not dist.is_initialized():
            return
        case = "baseline"

        train_sampler, test_sampler = get_samplers(self.dataset, case, self.seed)

        self.assertEqual(train_sampler.__class__.__name__, "DistributedSampler")
        self.assertEqual(test_sampler.__class__.__name__, "DistributedSampler")

    def test_get_sampler_custom(self):
        if not dist.is_initialized():
            return
        case = "pre_step_local"

        train_sampler, test_sampler = get_samplers(self.dataset, case, self.seed)

        self.assertEqual(train_sampler.__class__.__name__, "CustomDistributedSampler")
        self.assertEqual(test_sampler.__class__.__name__, "DistributedSampler")

    def test_get_sampler_fail(self):
        if not dist.is_initialized():
            return
        case = "abcde"

        with self.assertRaises(AssertionError):
            train_sampler, test_sampler = get_samplers(self.dataset, case, self.seed)


class TestGetOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MagicMock()
        self.model.parameters.return_value = [torch.tensor(1.0)]

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_optimizer_sgd(self, dist_mock):
        run_config = {
            "optimizer": {
                "name": "sgd",
                "kwargs": {
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                },
            },
            "schedulers": {"reference-kn": 128},
            "batch-size": 32,
        }
        optimizer = get_optimizer(self.model, run_config)
        self.assertEqual(optimizer.__class__.__name__, "SGD")
        # Check if the optimizer is correctly initialized
        self.assertEqual(optimizer.defaults["lr"], 0.1)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0001)

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_optimizer_adam(self, dist_mock):
        run_config = {
            "optimizer": {
                "name": "adam",
                "kwargs": {
                    "lr": 0.1,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0001,
                },
            },
            "schedulers": {"reference-kn": 128},
            "batch-size": 32,
        }
        optimizer = get_optimizer(self.model, run_config)
        self.assertEqual(optimizer.__class__.__name__, "Adam")
        # Check if the optimizer is correctly initialized
        self.assertEqual(optimizer.defaults["lr"], 0.1)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["eps"], 1e-08)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0001)

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_optimizer_rmsprop(self, dist_mock):
        run_config = {
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
            "schedulers": {"reference-kn": 128},
            "batch-size": 32,
        }
        optimizer = get_optimizer(self.model, run_config)
        self.assertEqual(optimizer.__class__.__name__, "RMSprop")
        # Check if the optimizer is correctly initialized
        self.assertEqual(optimizer.defaults["lr"], 0.1)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0001)
        self.assertEqual(optimizer.defaults["alpha"], 0.99)
        self.assertEqual(optimizer.defaults["eps"], 1e-08)

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_optimizer_unknown(self, dist_mock):
        run_config = {
            "optimizer": {
                "name": "unknown",
                "kwargs": {
                    "lr": 0.1,
                },
            },
            "schedulers": {"reference-kn": 128},
            "batch-size": 32,
        }
        with self.assertRaises(NotImplementedError):
            get_optimizer(self.model, run_config)


class TestGetScheduler(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.model.parameters.return_value = [torch.tensor(1.0)]
        self.mock_optimizer = optim.SGD(self.model.parameters(), lr=0.1)

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scheduler_step(self, dist_mock):
        mock_run_config = {
            "schedulers": {
                "name": "step",
                "kwargs": {"gamma": 0.1, "step_size": 30},
                "warmup-epochs": 5,
                "reference-kn": 128,
            },
            "optimizer": {"kwargs": {"lr": 0.1}},
            "batch-size": 128,
        }

        scheduler = get_scheduler(self.mock_optimizer, mock_run_config)

        self.assertIsInstance(scheduler, SequentialLR)
        self.assertIsInstance(scheduler._schedulers[0], LinearLR)
        self.assertIsInstance(scheduler._schedulers[1], StepLR)

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scheduler_none(self, dist_mock):
        mock_run_config = {
            "schedulers": {
                "name": "none",
                "kwargs": {"gamma": 0.1, "step_size": 30},
                "reference-kn": 128,
                "warmup-epochs": 5,
            },
            "optimizer": {"kwargs": {"lr": 0.1}},
            "batch-size": 8,
        }

        scheduler = get_scheduler(self.mock_optimizer, mock_run_config)

        self.assertIsInstance(scheduler, LinearLR)
        self.assertEqual(scheduler.start_factor, 1.0)

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scheduler_not_implemented(self, dist_mock):
        mock_run_config = {
            "schedulers": {
                "name": "unknown",
                "kwargs": {"gamma": 0.1, "step_size": 30},
                "reference-kn": 128,
                "warmup-epochs": 5,
            },
            "optimizer": {"kwargs": {"lr": 0.1}},
            "batch-size": 128,
        }

        with self.assertRaises(NotImplementedError):
            get_scheduler(self.mock_optimizer, mock_run_config)

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scaled_lr_neutral(self, dist_mock):
        mock_run_config = {
            "batch-size": 32,
            "schedulers": {
                "reference-kn": 128,
            },
        }
        initial_lr = 0.1
        # initial_lr * batch_size * world_size / reference_kn
        expected_lr = initial_lr * 32 * 4 / 128

        self.assertEqual(expected_lr, get_scaled_lr(initial_lr, mock_run_config))

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scaled_lr_increase(self, dist_mock):
        mock_run_config = {
            "batch-size": 128,
            "schedulers": {
                "reference-kn": 128,
            },
        }
        initial_lr = 0.1
        # initial_lr * batch_size * world_size / reference_kn
        expected_lr = initial_lr * 128 * 4 / 128

        self.assertEqual(expected_lr, get_scaled_lr(initial_lr, mock_run_config))

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scaled_lr_decrease(self, dist_mock):
        mock_run_config = {
            "batch-size": 8,
            "schedulers": {
                "reference-kn": 128,
            },
        }
        initial_lr = 0.1
        # initial_lr * batch_size * world_size / reference_kn
        expected_lr = initial_lr * 8 * 4 / 128

        self.assertEqual(expected_lr, get_scaled_lr(initial_lr, mock_run_config))


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
            "optimizer": {"name": "sgd", "kwargs": {"lr": 0.1, "momentum": 0.9}},
            "batch-size": 128,
        }

    @patch("src.main.dist.get_world_size", return_value=4)
    def test_get_scheduler(self, mock_dist):
        self.__set_up_get_scheduler()
        scheduler = get_scheduler(self.optimizer, self.run_config)
        print(self.run_config["optimizer"]["kwargs"]["lr"])
        assert isinstance(scheduler, SequentialLR)
        assert len(scheduler._schedulers) == 2
        assert isinstance(scheduler._schedulers[0], LinearLR)
        assert isinstance(scheduler._schedulers[1], StepLR)


if __name__ == "__main__":
    unittest.main()
