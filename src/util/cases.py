import re

from src.data.partition import SequentialPartitioner, StepPartitioner


class Case:
    def __init__(self, name, pre_shuffle, partitioner, shuffle, adjusted=False):
        self.name = name
        self.pre_shuffle = pre_shuffle
        self.partitioner = partitioner
        self.shuffle = shuffle
        self.adjusted = adjusted


class CaseFactory:
    @staticmethod
    def create_case(name: str) -> Case:
        """
        Create a configuration from a name.
        Expect the name to be of the form "(pre|asis)_(step|seq)_(local|noshuffle)(_(adj))?".
        Adjusted indicates, that a shuffle operation is performed after the partitioning.
        Independent of using local or noshuffle.
        :param name: Name of the configuration.
        :return: Configuration object.
        """
        match = re.fullmatch(r"(pre|asis)_(step|seq)_(local|noshuffle)(_(adj))?", name)
        # Throw expressive error message when name is invalid (match is None).
        assert match is not None, (
            f"Invalid name '{name}', "
            f"expected name to be of the form '(pre|asis)_(step|seq)_(local|noshuffle)(_(adj))?'"
        )
        pre_shuffle = match.group(1) == "pre"
        partitioner = (
            StepPartitioner() if match.group(2) == "step" else SequentialPartitioner()
        )
        shuffle = match.group(3) == "local"
        # Check if group 4 is present, if yes then the case is adjusted.
        adjusted = match.group(4) is not None
        return Case(name, pre_shuffle, partitioner, shuffle, adjusted)
