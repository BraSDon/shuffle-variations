import os
from time import time
import wandb

import torch
import torch.distributed as dist
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
            self.test(epoch)

    def run_epoch(self, epoch: int):
        self.model.train()

        loss_sum = 0
        acc1_sum = 0
        acc5_sum = 0
        num_batches = len(self.train_loader)
        start_time = time()
        # TODO: Log label distribution and/or JSD between batch and dataset
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()
            top1, top5 = self.calculate_accuracy(outputs, labels)
            acc1_sum += top1
            acc5_sum += top5

        required_time = time() - start_time
        self.log_statistics(
            acc1_sum, acc5_sum, epoch, loss_sum, num_batches, required_time, train=True
        )

        print0(f"Epoch {epoch} finished")

    def log_statistics(
        self, acc1_sum, acc5_sum, epoch, loss_sum, num_batches, required_time, train
    ):
        loss_avg_local = loss_sum / num_batches
        acc1_avg_local = acc1_sum / num_batches
        acc5_avg_local = acc5_sum / num_batches

        prefix = "train" if train else "test"

        wandb.log(
            {
                "epoch": epoch,
                f"{prefix}_epoch_time": required_time,
                f"{prefix}_loss": loss_avg_local,
                f"{prefix}_acc1": acc1_avg_local,
                f"{prefix}_acc5": acc5_avg_local,
            }
        )
        loss_avg = self.average_statistic(loss_avg_local)
        acc1_avg = self.average_statistic(acc1_avg_local)
        acc5_avg = self.average_statistic(acc5_avg_local)
        wandb.log(
            {
                "epoch": epoch,
                f"{prefix}_loss_avg": loss_avg,
                f"{prefix}_acc1_avg": acc1_avg,
                f"{prefix}_acc5_avg": acc5_avg,
            }
        )

    def test(self, epoch: int):
        self.model.eval()

        loss_sum = 0
        acc1_sum = 0
        acc5_sum = 0
        num_batches = len(self.test_loader)
        start_time = time()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss_sum += loss.item()
                top1, top5 = self.calculate_accuracy(outputs, labels)
                acc1_sum += top1
                acc5_sum += top5
        required_time = time() - start_time
        self.log_statistics(
            acc1_sum, acc5_sum, epoch, loss_sum, num_batches, required_time, train=False
        )

    def average_statistic(self, statistic):
        loss_tensor = torch.tensor(statistic).to(self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return loss_tensor.item() / dist.get_world_size()

    def calculate_accuracy(self, outputs, labels, topk=(1, 5)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = labels.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return res


def print0(message):
    if dist.get_rank() == 0:
        print(message)
