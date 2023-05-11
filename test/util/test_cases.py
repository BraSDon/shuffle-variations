import unittest
from src.data.partition import SequentialPartitioner, StepPartitioner
from src.util.cases import Case, CaseFactory


class TestCaseFactory(unittest.TestCase):
    def test_create_case_pre_step_local(self):
        # Arrange
        name = "pre_step_local"
        expected_case = Case(name, True, StepPartitioner(), True)

        # Act
        actual_case = CaseFactory.create_case(name)

        # Assert
        self.assertEqual(actual_case.name, expected_case.name)
        self.assertEqual(actual_case.pre_shuffle, expected_case.pre_shuffle)
        self.assertIsInstance(actual_case.partitioner, StepPartitioner)
        self.assertEqual(actual_case.shuffle, expected_case.shuffle)

    def test_create_case_asis_seq_noshuffle(self):
        # Arrange
        name = "asis_seq_noshuffle"
        expected_case = Case(name, False, SequentialPartitioner(), False)

        # Act
        actual_case = CaseFactory.create_case(name)

        # Assert
        self.assertEqual(actual_case.name, expected_case.name)
        self.assertEqual(actual_case.pre_shuffle, expected_case.pre_shuffle)
        self.assertIsInstance(actual_case.partitioner, SequentialPartitioner)
        self.assertEqual(actual_case.shuffle, expected_case.shuffle)

    def test_create_case_invalid_name(self):
        # Arrange
        name = "pre_step"

        # Act & Assert
        with self.assertRaises(AssertionError):
            CaseFactory.create_case(name)


if __name__ == "__main__":
    unittest.main()
