import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset, Subset

from src.util.cases import Case


class CustomDistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, case: Case, seed: int = 0) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.case = case

        # Make dataset evenly divisible by world_size.
        self.total_size = len(dataset) - (len(dataset) % self.world_size)
        self.dataset = Subset(dataset, range(self.total_size))

        self.seed = seed + self.rank
        self.generator = torch.Generator()
        # Each rank receives different generator seed.
        self.generator.manual_seed(self.seed)

        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"Invalid rank {self.rank}, "
                f"rank should be in the interval [0, {self.world_size - 1}]"
            )
        if self.case.pre_shuffle:
            # TODO: Test if all ranks get the same indices.
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        self.indices = self.case.partitioner.partition(self.world_size, indices)[
            self.rank
        ]

    def __iter__(self):
        if self.case.shuffle:
            indices = torch.randperm(
                len(self.dataset), generator=self.generator
            ).tolist()
        else:
            indices = self.indices

        return iter(indices)

    def __len__(self):
        return len(self.indices)
