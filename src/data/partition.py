from abc import abstractmethod, ABCMeta


class Partitioner(metaclass=ABCMeta):
    @abstractmethod
    def partition(self, world_size: int, indices: list[int]) -> list[list[int]]:
        """Partitions indices into world_size disjoint partitions of the same size."""
        pass


class SequentialPartitioner(Partitioner):
    def partition(self, world_size, indices):
        len_per_rank = len(indices) // world_size
        return [
            indices[i * len_per_rank : (i + 1) * len_per_rank]
            for i in range(world_size)
        ]


class StepPartitioner(Partitioner):
    def partition(self, world_size, indices):
        return [indices[rank::world_size] for rank in range(world_size)]
