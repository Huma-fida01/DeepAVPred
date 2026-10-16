"""Evaluation utilities for DeepAVPred predictions."""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


def _validated_labels_and_scores(labels, probabilities):
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if labels.shape[0] != probabilities.shape[0]:
        raise ValueError("Label and probability counts differ")
    if labels.shape[0] == 0:
        raise ValueError("Evaluation arrays must not be empty")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("Labels must contain only 0 and 1")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must be between 0 and 1")
    return labels.astype(int), probabilities


def compute_binary_metrics(labels, probabilities, threshold=0.5):
    """Calculate standard metrics from true labels and positive probabilities.

    The default threshold is 0.5 and may be changed by the caller. The
    manuscript does not specify a classification threshold.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    labels, probabilities = _validated_labels_and_scores(labels, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp) / (tp + fn) if tp + fn else float("nan")
    specificity = float(tn) / (tn + fp) if tn + fp else float("nan")
    npv = float(tn) / (tn + fn) if tn + fn else float("nan")
    roc_auc = roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2 else float("nan")
    average_precision = (
        average_precision_score(labels, probabilities)
        if len(np.unique(labels)) == 2 else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": float(roc_auc),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "npv": npv,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision),
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def load_array(path):
    if not os.path.isfile(path):
        raise IOError("Required input file does not exist: {0}".format(path))
    if path.lower().endswith(".npy"):
        return np.load(path, allow_pickle=False)
    return np.loadtxt(path, delimiter=",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--probabilities", help="Existing positive-class probabilities")
    parser.add_argument("--features", help="Existing 1040-column features")
    parser.add_argument("--model", help="Saved reconstructed Keras model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.probabilities:
        probabilities = load_array(args.probabilities)
    elif args.features and args.model:
        import tensorflow as tf
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_dir = os.path.join(project_root, "04_models", "DeepAVPred")
        sys.path.insert(0, model_dir)
        from attention_layer import TemporalAttention
        from deepavpred_model import prepare_input_matrix
        features = prepare_input_matrix(load_array(args.features))
        model = tf.keras.models.load_model(
            args.model, custom_objects={"TemporalAttention": TemporalAttention}
        )
        probabilities = model.predict(features, verbose=0).reshape(-1)
    else:
        raise ValueError(
            "Supply --probabilities, or supply both --features and --model. "
            "Both are required for direct model evaluation."
        )

    report = compute_binary_metrics(
        load_array(args.labels), probabilities, threshold=args.threshold
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
