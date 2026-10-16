"""Keras Tuner configuration for DeepAVPred.

The search uses the ranges documented in Supplementary Table S3. Random Search,
discretization, validation split, objective, trial count, and seed are
configurable implementation settings.
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import keras_tuner as kt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "04_models", "DeepAVPred")
sys.path.insert(0, MODEL_DIR)

from deepavpred_model import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    INPUT_DIMENSION,
    MAX_EPOCHS,
    build_deepavpred,
    prepare_input_matrix,
)


def load_array(path):
    """Load an existing NumPy or numeric CSV array."""
    if not os.path.isfile(path):
        raise IOError("Required input file does not exist: {0}".format(path))
    if path.lower().endswith(".npy"):
        return np.load(path, allow_pickle=False)
    return np.loadtxt(path, delimiter=",")


def validate_labels(labels, sample_count):
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != sample_count:
        raise ValueError("Feature and label sample counts differ")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("Labels must contain only 0 and 1")
    return labels.astype(np.float32)


def build_hypermodel(hp):
    return build_deepavpred(
        first_bigru_units=hp.Int("first_bigru_units", 64, 256, step=32),
        second_bigru_units=hp.Int("second_bigru_units", 32, 128, step=32),
        dropout_rate=hp.Float("dropout_rate", 0.2, 0.5, step=0.1),
        learning_rate=hp.Float("learning_rate", 1e-5, 1e-2, sampling="log"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", required=True, help="Existing 1040-column .npy or CSV"
    )
    parser.add_argument("--labels", required=True, help="Existing binary-label .npy or CSV")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "tuner"))
    parser.add_argument("--max-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = prepare_input_matrix(load_array(args.features))
    labels = validate_labels(load_array(args.labels), features.shape[0])
    if features.shape[1] != INPUT_DIMENSION:
        raise RuntimeError("Validated input dimension changed unexpectedly")

    tuner = kt.RandomSearch(
        build_hypermodel,
        objective=kt.Objective("val_roc_auc", direction="max"),
        max_trials=args.max_trials,
        seed=args.seed,
        directory=args.output_dir,
        project_name="deepavpred_reconstructed",
        overwrite=False,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    tuner.search(
        features,
        labels,
        validation_split=0.2,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )
    tuner.results_summary()


if __name__ == "__main__":
    main()
