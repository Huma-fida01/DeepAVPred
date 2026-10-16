# Public release notes

This repository is a **documented, manuscript-based reconstruction and partial implementation** of DeepAVPred, not a verified copy of the complete original training pipeline. See `CODE_PROVENANCE.md`.

## Included
- RECM (400-dimensional) implementation and tests.
- Attention-based BiGRU model, training/tuning and evaluation scripts.
- Archived ROC/PR figures and plotting notebook.

## Not included / not independently reproduced
- Raw sequence-to-RECM-DWT (1040D), CPSR, and ExPseAAC generators.
- Historical feature matrices, trained weights, exact fold assignments, and original hyperparameter selections.
- CSV inputs required by the figure notebook.
- AVPpred benchmark FASTA files (redistribution rights not confirmed).

The available code **does not independently reproduce the manuscript's reported performance** from raw sequences. Users must provide legitimate 1040-dimensional feature matrices for training. No public software license is granted in this package pending confirmation of contributor and institutional rights. Please obtain author approval before publication.
