import argparse
import os
import random
import sys

import yaml
import numpy as np
import torch
from torch.utils.data import Sampler, DistributedSampler
import torch.distributed as dist

sys.path.insert(0, sys.path[0] + "/../")

from src.data.data import MyDataset
from src.training.custom_sampler import CustomDistributedSampler
from src.util.cases import CaseFactory


def main():
    # 1. Parse arguments
    # 2. Setup distributed training (if applicable)
    # 2a. Set seeds
    # 3. Setup dataloaders
    # 4. Setup model
    # 5. Setup training_objects (criterion, optimizer, lr_scheduler)
    # 6. Setup trainer
    # 7. Run training
    # 8. Store results
    # 9. Free resources
    system_config, run_config = parse_configs()
    setup_distributed_training(
        system_config["system"], str(system_config["ddp"]["port"])
    )
    set_seeds(run_config["seed"])

    dataset = get_dataset(system_config, run_config)
    train_sampler, test_sampler = get_samplers(
        dataset, run_config["case"], run_config["seed"]
    )

    batch_size = run_config["batch-size"]
    num_workers = run_config["num-workers"]
    train_loader = dataset.get_train_loader(train_sampler, batch_size, num_workers)
    test_loader = dataset.get_test_loader(test_sampler, batch_size, num_workers)
    print(train_loader, test_loader)
    sanity_check()

    destroy_distributed_training()


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


def get_model(model: str) -> torch.nn.Module:
    """Return the model with the given name."""
    pass


def sanity_check():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank: {rank}, world_size: {world_size}")
    dist.barrier()


def destroy_distributed_training():
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
