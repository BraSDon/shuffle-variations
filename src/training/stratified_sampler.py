import torch.distributed as dist
from torch.utils.data import Sampler
from sklearn.model_selection import StratifiedShuffleSplit


class StratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=0):
        self.dataset = dataset
        self.seed = seed
        self.global_batch_size = batch_size * dist.get_world_size()
        self.targets = dataset.targets

        self.indices = self.__get_indices()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __get_indices(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        avail_indices = list(range(len(self.dataset)))
        sampled_indices = []
        while len(avail_indices) > 0:
            sampled, avail_indices = self.__sample(
                avail_indices, self.global_batch_size
            )
            sampled_indices.extend(sampled[rank::world_size])

        return sampled_indices

    def __sample(self, avail_indices, size) -> tuple[list[int], list[int]]:
        if size >= len(avail_indices):
            return avail_indices, []
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=size, random_state=self.seed
        )
        avail_targets = [self.targets[i] for i in avail_indices]
        sampled_indices, avail_indices = next(sss.split(avail_indices, avail_targets))
        return sampled_indices, avail_indices
