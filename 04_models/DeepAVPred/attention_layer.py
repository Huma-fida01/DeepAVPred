"""Temporal attention component for the DeepAVPred BiGRU stack.

The manuscript places attention between two BiGRU layers without specifying
its equation. This implementation learns one scalar score per timestep,
normalizes scores across time, and returns the reweighted sequence for the
second BiGRU.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="DeepAVPredReconstructed")
class TemporalAttention(tf.keras.layers.Layer):
    """Trainable scalar attention that preserves the input sequence shape."""

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("TemporalAttention expects rank-3 input")
        feature_count = int(input_shape[-1])
        self.score_kernel = self.add_weight(
            name="score_kernel",
            shape=(feature_count, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.score_bias = self.add_weight(
            name="score_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        scores = tf.tanh(tf.matmul(inputs, self.score_kernel) + self.score_bias)
        if mask is not None:
            expanded_mask = tf.cast(tf.expand_dims(mask, axis=-1), scores.dtype)
            scores = scores + (1.0 - expanded_mask) * tf.cast(-1e9, scores.dtype)
        weights = tf.nn.softmax(scores, axis=1)
        return inputs * weights

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        return super(TemporalAttention, self).get_config()
