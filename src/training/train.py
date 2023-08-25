from time import time
import wandb
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div

import torchmetrics
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.util.helper import print0


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        test_loader,
        system,
        my_dataset,
        device,
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.my_dataset = my_dataset
        self.device = device

        if system == "local":
            self.model = DDP(model.to(self.device))
        else:
            self.model = DDP(model.to(self.device), device_ids=[self.device])

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self.run_epoch(epoch)
            self.test(epoch)
            if self.scheduler is not None:
                self.scheduler.step()

    def run_epoch(self, epoch: int):
        self.model.train()

        loss_sum = 0.0
        acc1_sum = 0.0
        acc5_sum = 0.0
        mcc_sum = 0.0
        num_classes = self.my_dataset.num_classes
        num_batches = len(self.train_loader)
        label_frequencies = torch.zeros(self.my_dataset.num_classes, device=self.device)
        start_time = time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            acc1_sum, acc5_sum, loss_sum, mcc_sum = self.update_sums(
                acc1_sum,
                acc5_sum,
                labels,
                loss,
                loss_sum,
                mcc_sum,
                num_classes,
                outputs,
            )
            label_frequencies += torch.bincount(
                labels, minlength=self.my_dataset.num_classes
            )

            self.log_minibatch(labels, i, train=True)

        required_time = time() - start_time

        # Log the kl/js divergence of partition to full dataset.
        # Only required to be logged once
        if epoch == 0:
            js, kl, mean, std, _ = self._local_minibatch_statistics(label_frequencies)
            wandb.log(
                {
                    f"kl_div_rank{dist.get_rank()}": kl,
                    f"js_div_rank{dist.get_rank()}": js,
                    f"mean_error_rank{dist.get_rank()}": mean,
                    f"std_error_rank{dist.get_rank()}": std,
                }
            )
        self.log_statistics(
            acc1_sum,
            acc5_sum,
            mcc_sum,
            epoch,
            loss_sum,
            num_batches,
            required_time,
            train=True,
        )

        print0(f"Epoch {epoch} finished")

    def _local_minibatch_statistics(self, label_frequencies):
        label_frequencies = label_frequencies.float() / label_frequencies.sum()
        label_frequencies = label_frequencies.cpu()
        ref_freq = self.my_dataset.train_label_frequencies
        kl = sum(kl_div(label_frequencies, ref_freq))
        js = jensenshannon(label_frequencies, ref_freq)
        mean, std = self.calc_mean_std_error_of_frequency(label_frequencies, ref_freq)
        return js, kl, mean, std, label_frequencies

    def test(self, epoch: int):
        self.model.eval()

        loss_sum = 0.0
        acc1_sum = 0.0
        acc5_sum = 0.0
        mcc_sum = 0.0
        num_classes = self.my_dataset.num_classes
        num_batches = len(self.test_loader)
        start_time = time()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                acc1_sum, acc5_sum, loss_sum, mcc_sum = self.update_sums(
                    acc1_sum,
                    acc5_sum,
                    labels,
                    loss,
                    loss_sum,
                    mcc_sum,
                    num_classes,
                    outputs,
                )

                self.log_minibatch(labels, i, train=False)

        required_time = time() - start_time
        self.log_statistics(
            acc1_sum,
            acc5_sum,
            mcc_sum,
            epoch,
            loss_sum,
            num_batches,
            required_time,
            train=False,
        )

    @staticmethod
    def update_sums(
        acc1_sum, acc5_sum, labels, loss, loss_sum, mcc_sum, num_classes, outputs
    ):
        loss_sum += loss.item()
        acc1_sum += torchmetrics.functional.accuracy(
            outputs, labels, "multiclass", num_classes=num_classes
        )
        acc5_sum += torchmetrics.functional.accuracy(
            outputs,
            labels,
            "multiclass",
            num_classes=num_classes,
            top_k=min(5, num_classes),
        )
        mcc_sum += torchmetrics.functional.matthews_corrcoef(
            outputs, labels, "multiclass", num_classes=num_classes
        )
        return acc1_sum, acc5_sum, loss_sum, mcc_sum

    def log_statistics(
        self,
        acc1_sum,
        acc5_sum,
        mcc_sum,
        epoch,
        loss_sum,
        num_batches,
        required_time,
        train,
    ):
        loss_avg_local = loss_sum / num_batches
        acc1_avg_local = acc1_sum / num_batches
        acc5_avg_local = acc5_sum / num_batches
        mcc_avg_local = mcc_sum / num_batches

        prefix = "train" if train else "test"

        wandb.log(
            {
                "epoch": epoch,
                f"{prefix}_epoch_time": required_time,
                f"{prefix}_loss": loss_avg_local,
                f"{prefix}_acc1": acc1_avg_local,
                f"{prefix}_acc5": acc5_avg_local,
                f"{prefix}_mcc": mcc_avg_local,
            }
        )
        loss_avg = self.average_statistic(loss_avg_local)
        acc1_avg = self.average_statistic(acc1_avg_local)
        acc5_avg = self.average_statistic(acc5_avg_local)
        mcc_avg = self.average_statistic(mcc_avg_local)
        wandb.log(
            {
                "epoch": epoch,
                f"{prefix}_loss_avg": loss_avg,
                f"{prefix}_acc1_avg": acc1_avg,
                f"{prefix}_acc5_avg": acc5_avg,
                f"{prefix}_mcc_avg": mcc_avg,
            }
        )

    def average_statistic(self, statistic):
        if not isinstance(statistic, torch.Tensor):
            statistic = torch.tensor(statistic, device=self.device)
        else:
            statistic.to(self.device)
        dist.all_reduce(statistic, op=dist.ReduceOp.SUM)
        return statistic.item() / dist.get_world_size()

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
        label_frequencies_device = label_counts.float() / label_counts.sum()
        label_frequencies = label_frequencies_device.cpu()
        ref_freq = self.my_dataset.train_label_frequencies
        kl = sum(kl_div(label_frequencies, ref_freq))
        js = jensenshannon(label_frequencies, ref_freq)
        mean_error, std_error = self.calc_mean_std_error_of_frequency(
            label_frequencies, ref_freq
        )
        wandb.log(
            {
                f"{prefix}_batch": batch,
                f"{prefix}_local_label_frequencies": label_frequencies.tolist(),
                f"{prefix}_local_kl_div": kl,
                f"{prefix}_local_js_div": js,
                f"{prefix}_local_mean_error": mean_error,
                f"{prefix}_local_std_error": std_error,
            }
        )

        # Gather all label frequencies from all processes
        dist.all_reduce(label_frequencies_device, op=dist.ReduceOp.SUM)
        label_frequencies_device /= dist.get_world_size()
        label_frequencies = label_frequencies.cpu()
        kl = sum(kl_div(label_frequencies, ref_freq))
        js = jensenshannon(label_frequencies, ref_freq)
        mean_error, std_error = self.calc_mean_std_error_of_frequency(
            label_frequencies, ref_freq
        )
        wandb.log(
            {
                f"{prefix}_batch": batch,
                f"{prefix}_global_label_frequencies": label_frequencies.tolist(),
                f"{prefix}_global_kl_div": kl,
                f"{prefix}_global_js_div": js,
                f"{prefix}_global_mean_error": mean_error,
                f"{prefix}_global_std_error": std_error,
            }
        )

    @staticmethod
    def calc_mean_std_error_of_frequency(
        freq: torch.tensor, reference_freq: torch.tensor
    ):
        mean_error = torch.mean(torch.abs(freq - reference_freq))
        std_error = torch.std(torch.abs(freq - reference_freq))
        return mean_error, std_error
