from torch.optim.lr_scheduler import LRScheduler


class WarmupLR(LRScheduler):
    """
    Linear warmup learning rate scheduler.
    Implemented based on the strategy of: https://arxiv.org/abs/1706.02677.
    Warmup epochs are the number of epochs in which the learning rate is
    linearly increased.

    Example:
        >>> optimizer = torch.optim.SGD([torch.tensor(1.0)], lr=0.1)
        >>> scheduler = WarmupLR(optimizer, initial_lr=0.1, target_lr=1.1, warmup_epochs=5)
        >>> for epoch in range(6):
        >>>     print(optimizer.param_groups[0]['lr'])
        >>>     scheduler.step()
        0.1
        0.3
        0.5
        0.7
        0.9
        1.1
    """

    def __init__(
        self, optimizer, initial_lr, target_lr, warmup_epochs=5, verbose=False
    ):
        self.optimizer = optimizer
        self.step_size = (target_lr - initial_lr) / warmup_epochs
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        super().__init__(optimizer)

    def get_lr(self) -> list:
        epoch = self.last_epoch
        if 0 < epoch <= self.warmup_epochs:
            return [
                group["lr"] + self.step_size for group in self.optimizer.param_groups
            ]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]
