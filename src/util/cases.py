import re

from src.data.partition import SequentialPartitioner, StepPartitioner


class Case:
    def __init__(self, name, pre_shuffle, partitioner, shuffle):
        self.name = name
        self.pre_shuffle = pre_shuffle
        self.partitioner = partitioner
        self.shuffle = shuffle


class CaseFactory:
    @staticmethod
    def create_case(name: str) -> Case:
        """
        Create a configuration from a name.
        Expect the name to be of the form "{pre|asis}_{step|seq}_{local|noshuffle}".
        :param name: Name of the configuration.
        :return: Configuration object.
        """
        match = re.fullmatch(r"(pre|asis)_(step|seq)_(local|noshuffle)", name)
        # Throw expressive error message when name is invalid (match is None).
        assert match is not None, (
            f"Invalid name '{name}', "
            f"expected name to be of the form '{{pre|asis}}_{{step|seq}}_{{local|noshuffle}}'"
        )
        pre_shuffle = match.group(1) == "pre"
        partitioner = (
            StepPartitioner() if match.group(2) == "step" else SequentialPartitioner()
        )
        shuffle = match.group(3) == "local"
        return Case(name, pre_shuffle, partitioner, shuffle)
