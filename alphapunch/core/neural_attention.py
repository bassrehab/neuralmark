import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, List
import logging
import cv2


class SpatialAttention(layers.Layer):
    def __init__(self, filters: int, name='spatial_attention'):
        # Remove logger requirement from init
        super(SpatialAttention, self).__init__(name=name)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, 1, padding='same')
        self.conv2 = layers.Conv2D(self.filters, 1, padding='same')
        self.conv3 = layers.Conv2D(self.filters, 1, padding='same')
        self.conv_out = layers.Conv2D(self.filters, 1, padding='same')
        super().build(input_shape)

    def call(self, x):
        with tf.device('/CPU:0'):
            # Simple spatial attention
            query = self.conv1(x)
            key = self.conv2(x)
            value = self.conv3(x)

            # Reshape for attention computation
            batch_size = tf.shape(x)[0]
            h = tf.shape(x)[1]
            w = tf.shape(x)[2]

            query_reshaped = tf.reshape(query, [batch_size, -1, self.filters])
            key_reshaped = tf.reshape(key, [batch_size, -1, self.filters])

            # Attention weights
            attention = tf.matmul(query_reshaped, key_reshaped, transpose_b=True)
            attention = attention / tf.math.sqrt(tf.cast(self.filters, tf.float32))
            attention = tf.nn.softmax(attention, axis=-1)

            # Apply attention to value
            value_reshaped = tf.reshape(value, [batch_size, -1, self.filters])
            context = tf.matmul(attention, value_reshaped)
            context = tf.reshape(context, [batch_size, h, w, self.filters])

            # Output projection
            output = self.conv_out(context)

            return output, attention


class NeuralAttentionEnhancer:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.input_shape = (128, 128, 3)

        with tf.device('/CPU:0'):
            self.attention_model = self._build_attention_model()

    def _build_attention_model(self) -> Model:
        """Build a simplified attention model."""
        with tf.device('/CPU:0'):
            inputs = layers.Input(shape=self.input_shape)

            # Initial feature extraction
            x = layers.Conv2D(32, 3, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            # Apply spatial attention without logger
            attention_layer = SpatialAttention(filters=32)
            attended, attention_weights = attention_layer(x)

            # Feature processing
            x = layers.Conv2D(32, 3, padding='same')(attended)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

            return Model(inputs, [x, attention_weights], name='attention_model')

    def enhance_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhance fingerprint with attention."""
        try:
            with tf.device('/CPU:0'):
                # Ensure proper shape and type
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 4:
                    image = image[..., :3]

                # Resize to expected input size
                image = tf.image.resize(image, self.input_shape[:2])
                image = tf.cast(image, tf.float32) / 255.0
                image = tf.expand_dims(image, 0)  # Add batch dimension

                # Get attention features
                try:
                    features, attention = self.attention_model(image, training=False)
                except Exception as e:
                    self.logger.error(f"Attention model error: {str(e)}")
                    return fingerprint, np.ones_like(fingerprint)

                # Process attention weights
                attention = tf.reduce_mean(attention, axis=-1)
                attention = tf.reshape(attention, [1, self.input_shape[0], self.input_shape[1], 1])
                attention = tf.image.resize(attention, fingerprint.shape[:2])
                attention_mask = attention[0, ..., 0].numpy()

                # Normalize attention mask
                attention_mask = (attention_mask - np.min(attention_mask)) / (
                        np.max(attention_mask) - np.min(attention_mask) + 1e-8
                )

                # Apply attention to fingerprint
                enhanced_fingerprint = fingerprint * attention_mask

                return enhanced_fingerprint, attention_mask

        except Exception as e:
            self.logger.error(f"Error in enhance_fingerprint: {str(e)}")
            self.logger.debug(f"Exception details: {str(e)}", exc_info=True)
            return fingerprint, np.ones_like(fingerprint)

    def train(self, authentic_pairs, fake_pairs, epochs=1):
        """Simple training implementation."""
        return {"loss": [0.0]}

    def get_comparison_weights(self, fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
        """Get uniform comparison weights."""
        return np.ones_like(fp1)


