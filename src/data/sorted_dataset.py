import torch
import numpy as np
from torch.utils.data import Dataset


def sort_indices(dataset):
    labels = dataset.targets

    if isinstance(labels, torch.Tensor):
        sorted_idx = torch.argsort(labels)
    else:
        sorted_idx = np.argsort(labels)

    return sorted_idx


class SortedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sorted_indices = sort_indices(dataset)

    def __getitem__(self, index):
        return self.dataset[self.sorted_indices[index]]

    def __len__(self):
        return len(self.dataset)
