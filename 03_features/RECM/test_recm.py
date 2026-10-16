"""Tests for the DeepAVPred RECM implementation."""

from __future__ import division

import unittest

from recm import (
    AMINO_ACID_ORDER,
    RECM_ENERGY_MATRIX,
    recm_matrix,
    recm_vector,
)


class RecmTests(unittest.TestCase):

    def assertRowAlmostEqual(self, observed, expected, places=12):
        self.assertEqual(len(observed), len(expected))
        for actual, target in zip(observed, expected):
            self.assertAlmostEqual(actual, target, places=places)

    def test_aca_manuscript_example(self):
        matrix = recm_matrix("ACA")
        vector = recm_vector("ACA")
        a_index = AMINO_ACID_ORDER.index("A")
        c_index = AMINO_ACID_ORDER.index("C")

        self.assertEqual(len("ACA"), 3)
        self.assertEqual("ACA".count("A"), 2)
        self.assertEqual("ACA".count("C"), 1)
        self.assertRowAlmostEqual(
            matrix[a_index],
            tuple((2.0 / 3.0) * value for value in RECM_ENERGY_MATRIX[a_index]),
        )
        self.assertRowAlmostEqual(
            matrix[c_index],
            tuple((1.0 / 3.0) * value for value in RECM_ENERGY_MATRIX[c_index]),
        )

        zero_row = (0.0,) * 20
        for index, row in enumerate(matrix):
            if index not in (a_index, c_index):
                self.assertEqual(row, zero_row)

        self.assertEqual(len(matrix), 20)
        self.assertTrue(all(len(row) == 20 for row in matrix))
        self.assertEqual(len(vector), 400)

    def test_single_a_matches_source_energy_row(self):
        matrix = recm_matrix("A")
        a_index = AMINO_ACID_ORDER.index("A")
        self.assertRowAlmostEqual(matrix[a_index], RECM_ENERGY_MATRIX[a_index])
        for index, row in enumerate(matrix):
            if index != a_index:
                self.assertEqual(row, (0.0,) * 20)

    def test_output_dimensions_and_row_wise_flattening(self):
        matrix = recm_matrix("ACDEFGHIKLMNPQRSTVWY")
        vector = recm_vector("ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(len(matrix), 20)
        self.assertTrue(all(len(row) == 20 for row in matrix))
        self.assertEqual(len(vector), 400)
        self.assertEqual(vector[:20], matrix[0])
        self.assertEqual(vector[20:40], matrix[1])
        self.assertEqual(vector[-20:], matrix[-1])

    def test_invalid_residues_are_rejected(self):
        for sequence in ("AX", "AB", "AZ", "AU", "AO", "A-C", "A1", "A.C"):
            with self.assertRaises(ValueError):
                recm_vector(sequence)

    def test_empty_sequence_is_rejected(self):
        for sequence in ("", "   ", "\t\r\n"):
            with self.assertRaises(ValueError):
                recm_vector(sequence)

    def test_surrounding_whitespace_and_lowercase_usability_decisions(self):
        self.assertEqual(recm_vector("  aca\n"), recm_vector("ACA"))


if __name__ == "__main__":
    unittest.main()
