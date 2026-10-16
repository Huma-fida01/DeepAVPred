# Residue Energy Content Matrix (RECM)

This directory implements the RECM representation described in Section 4.2.3 of the DeepAVPred manuscript.

## Definition

Supplementary Table S2 provides a 20 x 20 residue-energy matrix using the following order for both axes:

```text
A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
```

For a peptide of length `L`, let `n_a` be the number of occurrences of amino-acid type `a`, and let `E_ab` be the corresponding value from Supplementary Table S2:

```text
R_ab = (n_a * E_ab) / L,  a,b = 1,...,20
```

Each energy-matrix row is therefore multiplied by the corresponding amino-acid frequency. Residue types absent from the sequence produce zero rows. Concatenating the 20 rows produces a 400-dimensional vector.

## API

- `recm_matrix(sequence)` returns the 20 x 20 sequence-specific matrix.
- `recm_vector(sequence)` returns the row-wise flattened 400-value vector.
- `flatten_recm(matrix)` validates and flattens a 20 x 20 matrix.

Input sequences are stripped of surrounding whitespace and converted to uppercase. Empty sequences and symbols outside the 20-residue alphabet are rejected.

## Validation

`test_recm.py` verifies the manuscript's `ACA` example, the single-residue `A` case, output dimensions, row-wise flattening, and invalid-input handling.

```bash
python test_recm.py -v
```

The implementation uses the equation and matrix values documented in the manuscript and Supplementary Table S2. Historical serialization formats and downstream preprocessing are not represented by these functions.

