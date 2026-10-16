# Code provenance and implementation notes

## 1. Evidence sources

The implementation follows the methodology and parameter information available in:

- `ijms-4553359.docx`
- `ijms-4553359-supplementary.docx`
- Surviving prediction, metric, and plotting artifacts in `HUMA_DeevAVPred_2026/`

Where the documents do not specify a software-level choice, the corresponding setting is documented as an implementation choice. The source documents and surviving project materials are maintained separately and are not modified by this package.

## 2. RECM implementation

The RECM equation, matrix construction, row concatenation, and 400-dimensional output follow Section 4.2.3 of the manuscript. The complete 20 x 20 energy matrix and amino-acid order come from Supplementary Table S2.

Input validation, lowercase conversion, immutable return structures, and runtime shape checks are software-interface choices. The Table S2 values and manuscript equation are used without additional scaling or normalization.

## 3. Feature representations

The manuscript documents CPSR, ExPseAAC, RECM, and RECM-DWT. Complete numerical definitions are available for RECM only.

- **CPSR:** the seven blocks and 71-dimensional total are documented, but the residue groups, property scales, calculations, normalization, and feature ordering are incomplete.
- **ExPseAAC:** the 20 + 25 dimensional structure and five property names are documented, but the property values, correlation equation, weighting, normalization, and complete ordering are unavailable.
- **RECM-DWT:** the manuscript documents 20 length-20 profiles, `bior3.3`, four decomposition levels, and a 1,040-dimensional total. The exact 52 features per profile and DWT boundary settings are not specified.

No unsupported sequence-level generators are included for these three representations. The model accepts an externally generated 1,040-column RECM-DWT matrix.

## 4. DeepAVPred model

The implemented layer order is based on the manuscript and Supplementary Table S3:

```text
BiGRU (64-256) -> Attention -> BiGRU (32-128) -> Dense(1, sigmoid)
```

Table S3 also provides Adam, learning rate `1e-5` to `1e-2`, L1/L2 coefficient `0.001`, dropout `0.2` to `0.5`, batch size 32, 200 epochs, patience 5, Keras Tuner, and binary cross-entropy.

The documents do not identify the selected range values, tensor reshape, attention equation, regularizer placement, recurrent dropout, tuner algorithm, or tuner objective. The code provides explicit, configurable settings for these items. The attention layer applies trainable temporal weighting while preserving the sequence for the second BiGRU.

## 5. Training and evaluation

The manuscript specifies 10-fold cross-validation on AVPtrain and separate AVPtest evaluation. `train_cv.py` generates stratified folds for each run because the original fold assignments and random seed are unavailable. Early stopping monitors validation loss and restores the best weights.

`evaluate.py` uses standard definitions for sensitivity, specificity, accuracy, MCC, ROC-AUC, precision, negative predictive value, F1, average precision, and confusion-matrix counts. The default classification threshold is 0.5 and can be changed from the command line.

## 6. Implementation notes

- Original tuner state, selected hyperparameters, trained weights, checkpoints, and training histories are not included in the surviving materials.
- Scientific training requires legitimate 1,040-column RECM-DWT matrices; the package does not create substitute feature data.
- The model's `(1040,) -> (1040, 1)` reshape is an explicit input convention used by this implementation.
- Software shape tests use synthetic arrays only to verify execution and do not produce biological results.
- Published performance values are not embedded as model targets or expected outputs.


## Public distribution
This repository contains a partial reconstruction. The original benchmark sequence files and historical artifacts are not all redistributed; availability of a public repository should not be interpreted as independent reproduction of the published results.
