import os
from time import time
import wandb
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.util.helper import print0


class Trainer:
    def __init__(
        self, model, optimizer, criterion, train_loader, test_loader, system, my_dataset
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.my_dataset = my_dataset

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
        for i, (inputs, labels) in enumerate(self.train_loader):
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

            self.log_minibatch(labels, i, train=True)

        required_time = time() - start_time
        self.log_statistics(
            acc1_sum, acc5_sum, epoch, loss_sum, num_batches, required_time, train=True
        )

        print0(f"Epoch {epoch} finished")

    def test(self, epoch: int):
        self.model.eval()

        loss_sum = 0
        acc1_sum = 0
        acc5_sum = 0
        num_batches = len(self.test_loader)
        start_time = time()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss_sum += loss.item()
                top1, top5 = self.calculate_accuracy(outputs, labels)
                acc1_sum += top1
                acc5_sum += top5

                self.log_minibatch(labels, i, train=False)

        required_time = time() - start_time
        self.log_statistics(
            acc1_sum, acc5_sum, epoch, loss_sum, num_batches, required_time, train=False
        )

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

    def average_statistic(self, statistic):
        tensor = torch.tensor(statistic).to(self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / dist.get_world_size()

    def calculate_accuracy(self, outputs, labels, topk=(1, 5)):
        with torch.no_grad():
            maxk = min(max(topk), self.my_dataset.num_classes)
            batch_size = labels.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return res

    def log_minibatch(
        self, local_minibatch_labels: torch.tensor, batch: int, train
    ) -> None:
        prefix = "train" if train else "test"

        # Calculate relative frequency of each label in the minibatch
        label_counts = torch.bincount(
            local_minibatch_labels, minlength=self.my_dataset.num_classes
        )
        label_frequencies = label_counts.float() / label_counts.sum()
        label_frequencies = label_frequencies.cpu()
        ref_freq = self.my_dataset.train_label_frequencies
        kl = sum(kl_div(label_frequencies, ref_freq))
        js = jensenshannon(label_frequencies, ref_freq)
        wandb.log(
            {
                f"{prefix}_batch": batch,
                f"{prefix}_local_label_frequencies": label_frequencies.tolist(),
                f"{prefix}_local_kl_div": kl,
                f"{prefix}_local_js_div": js,
            }
        )

        # Gather all label frequencies from all processes
        label_frequencies = label_frequencies.to(self.device)
        dist.all_reduce(label_frequencies, op=dist.ReduceOp.SUM)
        label_frequencies /= dist.get_world_size()
        label_frequencies = label_frequencies.cpu()
        kl = sum(kl_div(label_frequencies, ref_freq))
        js = jensenshannon(label_frequencies, ref_freq)
        wandb.log(
            {
                f"{prefix}_batch": batch,
                f"{prefix}_global_label_frequencies": label_frequencies.tolist(),
                f"{prefix}_global_kl_div": kl,
                f"{prefix}_global_js_div": js,
            }
        )
