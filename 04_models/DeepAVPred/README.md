# DeepAVPred attention-based BiGRU

Implementation of the architecture and parameter ranges described in the DeepAVPred manuscript and Supplementary Table S3.

## Purpose and input

The model performs binary antiviral-peptide classification from an existing RECM-DWT matrix:

```text
X.shape = (number_of_samples, 1040)
```

Raw sequence-to-RECM-DWT generation is outside this model module.

## Architecture

```text
1,040-dimensional feature input
-> Bidirectional GRU
-> Attention
-> Bidirectional GRU
-> Dense(1, sigmoid)
```

## Input reshape

The model reshapes each flat vector from `(1040,)` to `(1040, 1)`, treating each scalar feature as one timestep. This is the input convention used by this implementation; the manuscript does not define a tensor reshape.

## BiGRU and attention configuration

- First BiGRU units: 64-256, documented search range
- Second BiGRU units: 32-128, documented search range
- Default units: 128 and 64, configurable implementation settings
- Dropout: 0.2-0.5, documented search range; default 0.3
- Recurrent dropout: 0.0, implementation setting

The manuscript places attention between the two BiGRUs but does not provide its equation. `TemporalAttention` learns scalar timestep scores, applies a softmax across timesteps, and returns a reweighted sequence to the second BiGRU. The attention component was not independently ablated in the manuscript.

## Output, optimizer, and loss

- Output: `Dense(1, sigmoid)`
- Optimizer: Adam
- Learning-rate search range: `1e-5` to `1e-2`; default `1e-3`
- L1 coefficient: `0.001`
- L2 coefficient: `0.001`
- Loss: binary cross-entropy

The regularizer is applied to both GRU input kernels and the output Dense kernel in this implementation.

## Training settings

| Setting | Value | Source/status |
|---|---:|---|
| Batch size | 32 | Supplementary Table S3 |
| Maximum epochs | 200 | Supplementary Table S3 |
| Early-stopping patience | 5 | Supplementary Table S3 |
| CV folds | 10 | Manuscript |
| Early-stopping monitor | Validation loss | Implementation setting |
| Restore best weights | Enabled | Implementation setting |
| Default threshold | 0.5 | Implementation setting |

## Hyperparameter tuning

`tune_deepavpred.py` uses Keras Tuner over the documented ranges. Random Search, search steps, validation split, objective, trial count, and seed are configurable implementation settings because they are not fixed by Table S3.

## Software validation

Run a shape-only forward pass with synthetic zeros:

```bash
python deepavpred_model.py
```

This check validates layer execution and output shape only. Scientific training requires real 1,040-column feature matrices.

## Limitations

The package does not include historical RECM-DWT matrices, fold assignments, tuner state, selected hyperparameters, trained weights, or checkpoints. Model runs may therefore differ from reported manuscript results.

