import os
import wandb

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, test_loader, system):
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader

        if system == "local":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DDP(model.to(self.device))
        else:
            gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
            global_rank = int(os.environ["SLURM_PROCID"])
            self.device = global_rank % gpus_per_node
            self.model = DDP(model.to(self.device), device_ids=[self.device])

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self.run_epoch(epoch)
            self.validate()

    def run_epoch(self, epoch):
        self.model.train()

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            wandb.log({"loss": loss.item()})

        print(f"Epoch {epoch} finished")

    def validate(self):
        pass
