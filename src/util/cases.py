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
        components = name.split("_")
        assert len(components) == 3

        pre_shuffle = components[0] == "pre"
        partitioner = (
            StepPartitioner() if components[1] == "step" else SequentialPartitioner()
        )
        shuffle = components[2] == "local"
        return Case(name, pre_shuffle, partitioner, shuffle)
