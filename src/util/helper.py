import torch.distributed as dist


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)
