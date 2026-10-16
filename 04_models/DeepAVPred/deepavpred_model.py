"""Attention-based BiGRU model corresponding to the DeepAVPred manuscript.

This module accepts already-generated 1,040-dimensional RECM-DWT features.
It does not generate RECM-DWT features from peptide sequences.
"""

import numpy as np
import tensorflow as tf

try:
    from .attention_layer import TemporalAttention
except ImportError:
    from attention_layer import TemporalAttention


INPUT_DIMENSION = 1040
FIRST_BIGRU_RANGE = (64, 256)
SECOND_BIGRU_RANGE = (32, 128)
LEARNING_RATE_RANGE = (1e-5, 1e-2)
DROPOUT_RANGE = (0.2, 0.5)
L1_COEFFICIENT = 0.001
L2_COEFFICIENT = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 5


def prepare_input_matrix(features):
    """Validate an already-generated RECM-DWT matrix.

    Args:
        features: Numeric array with shape ``(n_samples, 1040)``.

    Returns:
        A finite ``float32`` NumPy array with the same two-dimensional shape.

    The model reshapes each flat vector to ``(1040, 1)`` and treats each scalar
    feature as one timestep. The manuscript does not define this tensor layout.
    """
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError("RECM-DWT features must be a rank-2 matrix")
    if matrix.shape[1] != INPUT_DIMENSION:
        raise ValueError(
            "Expected exactly {0} RECM-DWT features per sample; received {1}".format(
                INPUT_DIMENSION, matrix.shape[1]
            )
        )
    try:
        matrix = matrix.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        raise ValueError("RECM-DWT features must be numeric")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("RECM-DWT features contain NaN or infinite values")
    return matrix


def _validate_model_parameters(first_bigru_units, second_bigru_units, dropout_rate,
                               learning_rate):
    if not FIRST_BIGRU_RANGE[0] <= first_bigru_units <= FIRST_BIGRU_RANGE[1]:
        raise ValueError("first_bigru_units is outside the documented range")
    if not SECOND_BIGRU_RANGE[0] <= second_bigru_units <= SECOND_BIGRU_RANGE[1]:
        raise ValueError("second_bigru_units is outside the documented range")
    if not DROPOUT_RANGE[0] <= dropout_rate <= DROPOUT_RANGE[1]:
        raise ValueError("dropout_rate is outside the documented range")
    if not LEARNING_RATE_RANGE[0] <= learning_rate <= LEARNING_RATE_RANGE[1]:
        raise ValueError("learning_rate is outside the documented range")


def build_deepavpred(first_bigru_units=128, second_bigru_units=64,
                     dropout_rate=0.3, learning_rate=1e-3,
                     l1_coefficient=L1_COEFFICIENT,
                     l2_coefficient=L2_COEFFICIENT):
    """Build and compile the attention-based BiGRU.

    Default units, dropout, and learning rate are configurable values within
    the documented search ranges. L1/L2 coefficients, optimizer family, loss,
    and sigmoid output follow Supplementary Table S3.
    """
    _validate_model_parameters(
        first_bigru_units, second_bigru_units, dropout_rate, learning_rate
    )
    regularizer = tf.keras.regularizers.L1L2(
        l1=l1_coefficient, l2=l2_coefficient
    )

    flat_input = tf.keras.Input(shape=(INPUT_DIMENSION,), name="recm_dwt_features")
    sequence = tf.keras.layers.Reshape(
        (INPUT_DIMENSION, 1), name="reconstructed_feature_sequence"
    )(flat_input)
    sequence = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            first_bigru_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizer,
        ),
        name="bidirectional_gru_1",
    )(sequence)
    sequence = TemporalAttention(name="reconstructed_attention")(sequence)
    representation = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            second_bigru_units,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizer,
        ),
        name="bidirectional_gru_2",
    )(sequence)
    probability = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=regularizer,
        name="avp_probability",
    )(representation)

    model = tf.keras.Model(flat_input, probability, name="DeepAVPred_reconstructed")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="sensitivity"),
        ],
    )
    return model


def software_shape_check():
    """Build the model and run one synthetic forward pass for shape validation.

    The all-zero input is used only to check software execution and shapes. The
    function reports no biological performance and writes no artifacts.
    """
    model = build_deepavpred()
    synthetic = np.zeros((2, INPUT_DIMENSION), dtype=np.float32)
    output = model(synthetic, training=False)
    if tuple(output.shape) != (2, 1):
        raise RuntimeError("Unexpected model output shape: {0}".format(output.shape))
    return model, output


if __name__ == "__main__":
    checked_model, checked_output = software_shape_check()
    print("SYNTHETIC SOFTWARE SHAPE TEST PASSED")
    print("input_shape={0}".format(checked_model.input_shape))
    print("output_shape={0}".format(tuple(checked_output.shape)))
