import sys
import unittest

sys.path.insert(0, sys.path[0] + "/../../")

from src.data.partition import SequentialPartitioner, StepPartitioner


class MyTestCase(unittest.TestCase):
    def test_sequential(self):
        indices = list(range(10))
        world_size = 5
        partitions = SequentialPartitioner().partition(world_size, indices)

        self.assertEqual(len(partitions), world_size)
        self.assertEqual(len(partitions[0]), len(partitions[1]))
        self.assertEqual(partitions[0], [0, 1])
        self.assertEqual(partitions[4], [8, 9])

        indices = indices[::-1]
        partitions = SequentialPartitioner().partition(world_size, indices)

        self.assertEqual(partitions[0], [9, 8])
        self.assertEqual(partitions[4], [1, 0])

    def test_step(self):
        indices = list(range(10))
        world_size = 5
        partitions = StepPartitioner().partition(world_size, indices)

        self.assertEqual(len(partitions), world_size)
        self.assertEqual(len(partitions[0]), len(partitions[1]))
        self.assertEqual(partitions[0], [0, 5])
        self.assertEqual(partitions[4], [4, 9])

        indices = indices[::-1]
        partitions = StepPartitioner().partition(world_size, indices)

        self.assertEqual(partitions[0], [9, 4])
        self.assertEqual(partitions[4], [5, 0])


if __name__ == "__main__":
    unittest.main()
