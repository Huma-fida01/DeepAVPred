"""Residue Energy Content Matrix (RECM) encoding for DeepAVPred.

For a peptide of length L, the sequence-specific matrix is

    R[a, b] = n[a] * E[a, b] / L

where n[a] is the occurrence count of amino acid a and E is the 20 x 20
property/energy matrix from Supplementary Table S2. Rows are concatenated in
``AMINO_ACID_ORDER`` to produce a 400-value feature vector.
"""

from __future__ import division

from collections import Counter


AMINO_ACID_ORDER = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)


# Values and row/column order from Supplementary Table S2. No additional
# scaling or normalization is applied to this source matrix.
RECM_ENERGY_MATRIX = (
    (-1.65, -2.83,  1.16,  1.80, -3.73, -0.41,  1.90, -3.69,  0.49, -3.01, -2.08,  0.66,  1.54,  1.20,  0.98, -0.08,  0.46, -2.31,  0.32, -4.62),
    (-2.83, -39.58, -0.82, -0.53, -3.07, -2.96, -4.98,  0.34, -1.38, -2.15,  1.43, -4.18, -2.13, -2.91, -0.41, -2.33, -1.84, -0.16,  4.26, -4.46),
    ( 1.16, -0.82,  0.84,  1.97, -0.92,  0.88, -1.07,  0.68, -1.93,  0.23,  0.61,  0.32,  3.31,  2.67, -2.02,  0.91, -0.65,  0.94, -0.71,  0.90),
    ( 1.80, -0.53,  1.97,  1.45,  0.94,  1.31,  0.61,  1.30, -2.51,  1.14,  2.53,  0.20,  1.44,  0.10, -3.13,  0.81,  1.54,  0.12, -1.07,  1.29),
    (-3.73, -3.07, -0.92,  0.94, -11.25,  0.35, -3.57, -5.88, -0.82, -8.59, -5.34,  0.73,  0.32,  0.77, -0.40, -2.22,  0.11, -7.05, -7.09, -8.80),
    (-0.41, -2.96,  0.88,  1.31,  0.35, -0.20,  1.09, -0.65, -0.16, -0.55, -0.52, -0.32,  2.25,  1.11,  0.84,  0.71,  0.59, -0.38,  1.69, -1.90),
    ( 1.90, -4.98, -1.07,  0.61, -3.57,  1.09,  1.97, -0.71,  2.89, -0.86, -0.75,  1.84,  0.35,  2.64,  2.05,  0.82, -0.01,  0.27, -7.58, -3.20),
    (-3.69,  0.34,  0.68,  1.30, -5.88, -0.65, -0.71, -6.74, -0.01, -9.01, -3.62, -0.07,  0.12, -0.18,  0.19, -0.15,  0.63, -6.54, -3.78, -5.26),
    ( 0.49, -1.38, -1.93, -2.51, -0.82, -0.16,  2.89, -0.01,  1.24,  0.49,  1.61,  1.12,  0.51,  0.43,  2.34,  0.19, -1.11,  0.19,  0.02, -1.19),
    (-3.01, -2.15,  0.23,  1.14, -8.59, -0.55, -0.86, -9.01,  0.49, -6.37, -2.88,  0.97,  1.81, -0.58, -0.60, -0.41,  0.72, -5.43, -8.31, -4.90),
    (-2.08,  1.43,  0.61,  2.53, -5.34, -0.52, -0.75, -3.62,  1.61, -2.88, -6.49,  0.21,  0.75,  1.90,  2.09,  1.39,  0.63, -2.59, -6.88, -9.73),
    ( 0.66, -4.18,  0.32,  0.20,  0.73, -0.32,  1.84, -0.07,  1.12,  0.97,  0.21,  0.61,  1.15,  1.28,  1.08,  0.29,  0.46,  0.93, -0.74,  0.93),
    ( 1.54, -2.13,  3.31,  1.44,  0.32,  2.25,  0.35,  0.12,  0.51,  1.81,  0.75,  1.15, -0.42,  2.97,  1.06,  1.12,  1.65,  0.38, -2.06, -2.09),
    ( 1.20, -2.91,  2.67,  0.10,  0.77,  1.11,  2.64, -0.18,  0.43, -0.58,  1.90,  1.28,  2.97, -1.54,  0.91,  0.85, -0.07, -1.91, -0.76,  0.01),
    ( 0.98, -0.41, -2.02, -3.13, -0.40,  0.84,  2.05,  0.19,  2.34, -0.60,  2.09,  1.08,  1.06,  0.91,  0.21,  0.95,  0.98,  0.08, -5.89,  0.36),
    (-0.08, -2.33,  0.91,  0.81, -2.22,  0.71,  0.82, -0.15,  0.19, -0.41,  1.39,  0.29,  1.12,  0.85,  0.95, -0.48, -0.06,  0.13, -3.03, -0.82),
    ( 0.46, -1.84, -0.65,  1.54,  0.11,  0.59, -0.01,  0.63, -1.11,  0.72,  0.63,  0.46,  1.65, -0.07,  0.98, -0.06, -0.96,  1.14, -0.65, -0.37),
    (-2.31, -0.16,  0.94,  0.12, -7.05, -0.38,  0.27, -6.54,  0.19, -5.43, -2.59,  0.93,  0.38, -1.91,  0.08,  0.13,  1.14, -4.82, -2.13, -3.59),
    ( 0.32,  4.26, -0.71, -1.07, -7.09,  1.69, -7.58, -3.78,  0.02, -8.31, -6.88, -0.74, -2.06, -0.76, -5.89, -3.03, -0.65, -2.13, -1.73, -12.39),
    (-4.62, -4.46,  0.90,  1.29, -8.80, -1.90, -3.20, -5.26, -1.19, -4.90, -9.73,  0.93, -2.09,  0.01,  0.36, -0.82, -0.37, -3.59, -12.39, -2.68),
)


def _validate_source_matrix():
    """Fail fast if the transcribed supplementary matrix is not 20 x 20."""
    if len(AMINO_ACID_ORDER) != 20:
        raise RuntimeError("AMINO_ACID_ORDER must contain exactly 20 residues")
    if len(RECM_ENERGY_MATRIX) != 20:
        raise RuntimeError("RECM_ENERGY_MATRIX must contain exactly 20 rows")
    if any(len(row) != 20 for row in RECM_ENERGY_MATRIX):
        raise RuntimeError("Every RECM_ENERGY_MATRIX row must contain 20 values")


def validate_sequence(sequence):
    """Return a validated peptide in uppercase.

    Surrounding whitespace is stripped and lowercase letters are converted to
    uppercase. Only the 20 residues in ``AMINO_ACID_ORDER`` are accepted;
    unsupported characters are not silently removed.
    """
    if not isinstance(sequence, str):
        raise TypeError("Peptide sequence must be a string")

    peptide = sequence.strip().upper()
    if not peptide:
        raise ValueError("Peptide sequence is empty after stripping whitespace")

    invalid = sorted(set(peptide) - set(AMINO_ACID_ORDER))
    if invalid:
        raise ValueError(
            "Unsupported residue character(s): {0}. Allowed residues: {1}".format(
                ", ".join(repr(char) for char in invalid),
                "".join(AMINO_ACID_ORDER),
            )
        )
    return peptide


def recm_matrix(sequence):
    """Create the manuscript-defined 20 x 20 RECM matrix for one peptide.

    Args:
        sequence: Peptide sequence containing only the 20 amino acids supported
            by Supplementary Table S2. Surrounding whitespace is stripped and
            lowercase letters are converted to uppercase.

    Returns:
        A tuple of 20 row tuples. Both axes follow ``AMINO_ACID_ORDER``.
        Entry ``[a][b]`` is ``n_a * E_ab / L``.
    """
    peptide = validate_sequence(sequence)
    counts = Counter(peptide)
    length = len(peptide)

    matrix = tuple(
        tuple((counts[amino_acid] * energy) / length for energy in energy_row)
        for amino_acid, energy_row in zip(AMINO_ACID_ORDER, RECM_ENERGY_MATRIX)
    )

    if len(matrix) != 20 or any(len(row) != 20 for row in matrix):
        raise RuntimeError("Generated RECM matrix does not have shape (20, 20)")
    return matrix


def flatten_recm(matrix):
    """Concatenate RECM rows in row-major order into a 400-value tuple."""
    if len(matrix) != 20 or any(len(row) != 20 for row in matrix):
        raise ValueError("RECM matrix must have shape (20, 20)")
    vector = tuple(value for row in matrix for value in row)
    if len(vector) != 400:
        raise RuntimeError("Flattened RECM vector must contain 400 values")
    return vector


def recm_vector(sequence):
    """Encode one peptide as a row-wise flattened 400-value RECM vector."""
    return flatten_recm(recm_matrix(sequence))


_validate_source_matrix()
