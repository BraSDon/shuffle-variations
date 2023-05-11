import argparse
import os
import random

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import torch.distributed as dist

from src.data.data import MyDataset
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
    # sampler = get_sampler(run_config["case"])
    # dataloaders = get_dataloaders(system_config, run_config, sampler)
    # train_loader, test_loader = dataloaders["train"], dataloaders["test"]
    # model = get_model(run_config["model"])
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


def get_sampler(case: str) -> Sampler:
    """Returns a sampler based on the case."""
    case = CaseFactory.create_case(case)
    pass


def get_dataloaders(
    system_config: dict, run_config: dict, sampler
) -> dict[str, DataLoader]:
    dataset_name = run_config["dataset"]
    path = system_config["datasets"][dataset_name]["path"]
    transformations = system_config["datasets"][dataset_name]["transformations"]
    load_function = system_config["datasets"][dataset_name]["load_function"]

    dataset = MyDataset(dataset_name, path, transformations, load_function)

    batch_size = run_config["batch_size"]
    num_workers = run_config["num_workers"]

    return dataset.get_dataloaders(sampler, batch_size, num_workers)


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
