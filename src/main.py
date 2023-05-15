import argparse
import os
import random
import sys

import torchvision.models.alexnet
import wandb
import yaml
import numpy as np
import torch
from torch.utils.data import Sampler, DistributedSampler
import torch.distributed as dist

sys.path.insert(0, sys.path[0] + "/../")

from src.models.models import DummyModel
from src.training.train import Trainer
from src.data.data import MyDataset
from src.training.custom_sampler import CustomDistributedSampler
from src.util.cases import CaseFactory


def main():
    # 1. Parse arguments
    system_config, run_config = parse_configs()

    # 2. Setup distributed training (if applicable)
    setup_distributed_training(
        system_config["system"], str(system_config["ddp"]["port"])
    )

    # 2a. Setup logging
    setup_logging(run_config)

    # 3. Set seed
    set_seeds(run_config["seed"])

    # 4. Setup dataloaders
    dataset = get_dataset(system_config, run_config)
    train_sampler, test_sampler = get_samplers(
        dataset, run_config["case"], run_config["seed"]
    )
    batch_size = run_config["batch-size"]
    num_workers = run_config["num-workers"]
    train_loader = dataset.get_train_loader(train_sampler, batch_size, num_workers)
    test_loader = dataset.get_test_loader(test_sampler, batch_size, num_workers)

    # 5. Setup model
    model = get_model_by_name(run_config["model"])
    assert model is not None

    # 6. Setup training_objects (criterion, optimizer)
    criterion = get_criterion(run_config["criterion"])
    optimizer = get_optimizer(
        run_config["optimizer"],
        model,
        run_config["learning-rate"],
        run_config["weight-decay"],
        run_config["momentum"],
    )

    # 7. Perform sanity check before training
    sanity_check(system_config["system"])

    # 8. Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        system=system_config["system"],
    )

    # 9. Run training
    trainer.train(run_config["max-epochs"])

    # TODO: 10. Store results

    # 11. Free resources
    free_resources()


def parse_configs() -> tuple[dict, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="../run-configs/default-config.yaml"
    )

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        run_config = yaml.safe_load(f)

    with open("../system-config.yaml", "r") as f:
        system_config = yaml.safe_load(f)

    return system_config, run_config


def setup_distributed_training(system: str, port: str):
    if system == "local":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group(
            "gloo", rank=int(os.environ["LOCAL_RANK"]), world_size=4
        )
        return
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    addr = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def setup_logging(run_config):
    # TODO: Log to node local file system to avoid load on shared file system.
    wandb.init(
        project="paper",
        group=run_config["group"],
        name=f"{run_config['case']}-rank-{dist.get_rank()}",
        config=run_config,
    )


def set_seeds(seed: int):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_samplers(mydataset: MyDataset, case: str, seed: int) -> tuple[Sampler, Sampler]:
    """Returns a sampler based on the case."""
    case = CaseFactory.create_case(case)
    train_sampler = CustomDistributedSampler(mydataset.train_dataset, case, seed)
    test_sampler = DistributedSampler(mydataset.test_dataset)
    return train_sampler, test_sampler


def get_dataset(system_config: dict, run_config: dict) -> MyDataset:
    dataset_name = run_config["dataset"]
    path = system_config["datasets"][dataset_name]["path"]
    train_transformations = system_config["datasets"][dataset_name]["transforms"][
        "train"
    ]
    test_transformations = system_config["datasets"][dataset_name]["transforms"]["test"]
    load_function = system_config["datasets"][dataset_name]["load-function"]

    return MyDataset(
        dataset_name, path, train_transformations, test_transformations, load_function
    )


def get_model_by_name(model: str) -> torch.nn.Module:
    """Return the model with the given name."""
    if model == "dummy":
        return DummyModel()
    elif model == "alexnet":
        return torchvision.models.alexnet()
    else:
        raise NotImplementedError(f"Model {model} not implemented.")


def get_criterion(criterion: str):
    """Return the criterion with the given name."""
    if criterion == "cross-entropy":
        return torch.nn.CrossEntropyLoss()
    elif criterion == "mse":
        return torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"Criterion {criterion} not implemented.")


def get_optimizer(optimizer: str, model: torch.nn.Module, lr, weight_decay, momentum):
    """Return the optimizer with the given name."""
    if optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented.")


def sanity_check(device):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank: {rank}, world_size: {world_size}")
    dist.barrier()
    print(f"Running on {device}")


def free_resources():
    wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
