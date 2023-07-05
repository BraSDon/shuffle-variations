import importlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms

# Bug fix: https://discuss.pytorch.org/t/oserror-image-file-is-truncated-150-bytes-not-processed/64445
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.data.sorted_dataset import SortedDataset


class MyDataset:
    """Class for handling datasets and returning dataloaders."""

    def __init__(self, name: str, path: str, train_transformations: list[dict], test_transformations: list[dict],
                 load_function: dict, num_classes: int, device):
        self.name = name
        self.path = path
        self.num_classes = num_classes
        self.device = device

        self.train_transform = self._extract_transform(train_transformations)
        self.test_transform = self._extract_transform(test_transformations)
        self.load_function, self._lf_type = self._extract_load_function(load_function)

        # Move dataset to node-local storage.
        self.path = self.copy_dataset_to_tmp()

        self.train_dataset, self.test_dataset = self.__get_datasets()
        self.train_label_frequencies = self._get_train_label_frequencies()
        self.sort_train_dataset()  # Make sure that train_dataset is sorted by class.

    def get_train_loader(
        self, sampler: Sampler, batch_size: int, num_workers: int, **kwargs
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )

    def get_test_loader(
        self, sampler: Sampler, batch_size: int, num_workers: int, **kwargs
    ) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )

    def __get_datasets(self) -> tuple[Dataset, Dataset]:
        assert callable(self.load_function)
        # NOTE: Adjust if load-function requires it.
        if self._lf_type == "built-in":
            train_dataset = self.load_function(
                self.path, transform=self.train_transform, train=True
            )
            test_dataset = self.load_function(
                self.path, transform=self.test_transform, train=False
            )
        elif self._lf_type == "generic":
            train_dataset = self.load_function(
                f"{self.path}/train", transform=self.train_transform
            )
            test_dataset = self.load_function(
                f"{self.path}/test", transform=self.test_transform
            )
        # ImageNet is an exception, since the folder structure of test does
        # not meet the specifications, which is that it should have a subfolder
        # for each class. Because of time restrictions, we resort to using the
        # validation set as test set, as also done in official PyTorch examples.
        # see: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L245
        elif self._lf_type == "imagenet":
            train_dataset = self.load_function(
                f"{self.path}/train", transform=self.train_transform
            )
            test_dataset = self.load_function(
                f"{self.path}/val", transform=self.test_transform
            )
        else:
            raise ValueError(f"Unknown load function type: {self._lf_type}")
        return train_dataset, test_dataset

    def sort_train_dataset(self):
        start = time()
        self.train_dataset = SortedDataset(self.train_dataset)
        wandb.log({"train_dataset_sort_time": time() - start})

    def _get_train_label_frequencies(self):
        # Calculate label frequency for train dataset.
        start = time()
        target_tensor = torch.Tensor(self.train_dataset.targets).type(torch.int32)
        bincount = torch.bincount(target_tensor, minlength=self.num_classes)
        print(f"Train_label_freq_calc_time: {time() - start}")
        wandb.log({"train_label_freq_calc_time": time() - start})
        return bincount / bincount.sum()

    @staticmethod
    def _extract_transform(transformations: list[dict]) -> transforms.Compose:
        if transformations is None:
            return None
        transform_list = []
        for t in transformations:
            transform_class = getattr(transforms, t["name"])
            kwargs = t["kwargs"] if "kwargs" in t else {}
            transform_list.append(transform_class(**kwargs))
        return transforms.Compose(transform_list)

    @staticmethod
    def _extract_load_function(load_function: dict) -> tuple[callable, str]:
        module, lf_type, name = (
            load_function["module"],
            load_function["type"],
            load_function["name"],
        )

        # Import specified module and
        # get the function in that module with the specified name.
        module = importlib.import_module(module)
        load_function = getattr(module, name)
        return load_function, lf_type

    def copy_dataset_to_tmp(self):
        try:
            tmp_path = os.environ["TMPDIR"]
        except KeyError:
            print("TMPDIR not set, using original path.")
            return self.path
        src_dir = self.path
        dst_dir = f"{tmp_path}/{self.name}"
        global_rank = dist.get_rank()
        local_rank = global_rank % int(os.environ["SLURM_GPUS_ON_NODE"])
        if local_rank == 0:
            start = time()
            print(f"[GPU {global_rank}] Copying dataset to {dst_dir}...")
            self.copy_files(src_dir, dst_dir)
            print(f"[GPU {global_rank}] Done copying dataset in {time() - start:.2f}s.")
            wandb.log({"dataset_copy_time": time() - start})
        dist.barrier()
        return dst_dir

    def copy_files(self, src_dir, dst_dir, max_workers=32):
        os.makedirs(dst_dir, exist_ok=True)

        # Get a list of all files and directories in the source directory
        for dirpath, dirnames, filenames in os.walk(src_dir):
            # Create the corresponding subdirectories in the destination directory
            subdir = dirpath.replace(src_dir, dst_dir)
            os.makedirs(subdir, exist_ok=True)

            # Create a ThreadPoolExecutor with a max of 4 worker threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit the copy_file function for each file in the list
                futures = [
                    executor.submit(
                        self.copy_file,
                        os.path.join(dirpath, f),
                        os.path.join(subdir, f),
                    )
                    for f in filenames
                ]
                for future in as_completed(futures):
                    future.result()

    @staticmethod
    def copy_file(src, dst):
        shutil.copy2(src, dst)
