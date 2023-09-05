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
        # pre vs. asis
        if self.case.pre_shuffle:
            # Create a generator with the original seed, such that all ranks
            # perform the same pre-shuffle.
            common_generator = torch.Generator()
            common_generator.manual_seed(seed)
            self._pre_indices = torch.randperm(
                len(self.dataset), generator=common_generator
            ).tolist()
        else:
            self._pre_indices = list(range(len(self.dataset)))

        # step-wise vs. sequential partitioning
        self.indices = self.case.partitioner.partition(
            self.world_size, self._pre_indices
        )[self.rank]

    def __iter__(self):
        # local vs. no-shuffle
        if self.case.shuffle:
            permutation = torch.randperm(len(self.indices), generator=self.generator)
            indices = torch.gather(torch.tensor(self.indices), 0, permutation)
        else:
            indices = self.indices

        # TODO: Track indices for each rank. (should change wandb dir then)
        return iter(indices)

    def __len__(self):
        return len(self.indices)
