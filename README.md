# DeepAVPred

DeepAVPred is an attention-based bidirectional gated recurrent unit (BiGRU) framework for antiviral peptide prediction. This repository contains organized scripts corresponding to the feature representation, model architecture, training, and evaluation procedures described in the manuscript and supplementary material.

## Overview

The implemented workflow accepts a 1,040-dimensional RECM-DWT feature matrix and applies the following architecture:

```text
1,040-dimensional input
-> Bidirectional GRU
-> Attention
-> Bidirectional GRU
-> Dense sigmoid output
```

The repository also includes a validated implementation of the 400-dimensional Residue Energy Content Matrix (RECM) representation. Exact sequence-level generators are not included for CPSR, ExPseAAC, or RECM-DWT because their complete numerical definitions are not available in the project documentation.

## Dataset

The AVPtrain and AVPtest benchmark datasets used in the DeepAVPred study were adopted from the previously published AVPpred benchmark dataset described by Thakur et al. [13]. AVPtrain contains 1,088 sequences (544 AVPs and 544 non-AVPs) and was used for model training and 10-fold cross-validation. AVPtest contains 120 sequences (60 AVPs and 60 non-AVPs) and was used for separate performance evaluation.

**Reference [13]:** Thakur, N.; Qureshi, A.; Kumar, M. AVPpred: Collection and prediction of highly effective antiviral peptides. *Nucleic Acids Res.* 2012, 40, W199–W204. https://doi.org/10.1093/nar/gks450.

## Repository structure

```text
03_features/RECM/          RECM feature generator and unit tests
04_models/DeepAVPred/      BiGRU model and attention implementation
05_training/               Training and tuning scripts
06_evaluation/             Evaluation metrics
07_figures/                Archived figure images and plotting notebook
CODE_PROVENANCE.md         Evidence basis and implementation decisions
requirements.txt          Python dependencies
```

**Data access:** The AVPtrain and AVPtest benchmark files are not redistributed in this public-release package pending confirmation of redistribution permissions. See the AVPpred source citation above to locate the original benchmark. No original model weights or feature matrices are included.

## Feature representations

- **RECM:** implemented as a 20 x 20 matrix using the amino-acid order and energy values in Supplementary Table S2. Row-wise flattening produces 400 features.
- **RECM-DWT:** the model expects an existing 1,040-column matrix. Raw sequence-to-RECM-DWT generation is not included.
- **CPSR and ExPseAAC:** represented in the manuscript, but their complete numerical transformations are not implemented here.

## DeepAVPred architecture

The model configuration follows the layer order and parameter information in the manuscript and Supplementary Table S3. Search ranges are retained for both BiGRU layers, dropout, and learning rate. Software-level settings that are not fixed in the documents remain configurable and are described in the model README.

## Training

`05_training/train_cv.py` performs stratified 10-fold cross-validation on a supplied feature matrix and labels. `05_training/tune_deepavpred.py` defines a Keras Tuner search over the documented ranges. Both scripts require real input files and do not generate replacement feature data.

## Evaluation

`06_evaluation/evaluate.py` calculates accuracy, sensitivity, specificity, MCC, ROC-AUC, precision, negative predictive value, F1, average precision, and confusion-matrix counts.

## Requirements

Install the dependencies in an isolated Python environment:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the RECM tests:

```bash
cd 03_features/RECM
python test_recm.py -v
```

Validate the model structure with synthetic shape-test input:

```bash
python 04_models/DeepAVPred/deepavpred_model.py
```

Run cross-validation after supplying a legitimate 1,040-column matrix:

```bash
python 05_training/train_cv.py --features train_features.npy --labels train_labels.npy
```

Evaluate existing probabilities:

```bash
python 06_evaluation/evaluate.py --labels labels.npy --probabilities probabilities.npy
```

See [CODE_PROVENANCE.md](CODE_PROVENANCE.md) for the evidence basis and implementation notes. Running this package does not by itself guarantee reproduction of published performance because the historical feature matrices, fold assignments, selected hyperparameters, and trained weights are not available.
