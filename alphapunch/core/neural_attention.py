import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, List
import logging


class AttentionModule(layers.Layer):
    def __init__(self, filters: int):
        super(AttentionModule, self).__init__()
        self.filters = filters
        self.query_conv = layers.Conv2D(filters, 1)
        self.key_conv = layers.Conv2D(filters, 1)
        self.value_conv = layers.Conv2D(filters, 1)
        self.output_conv = layers.Conv2D(filters, 1)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        query = tf.reshape(query, [batch_size, -1, self.filters])
        key = tf.reshape(key, [batch_size, -1, self.filters])
        value = tf.reshape(value, [batch_size, -1, self.filters])

        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = attention_weights / tf.math.sqrt(tf.cast(self.filters, tf.float32))
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        attended = tf.matmul(attention_weights, value)
        attended = tf.reshape(attended, [batch_size, height, width, self.filters])
        output = self.output_conv(attended)

        return output, attention_weights


class NeuralAttentionEnhancer:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.input_shape = (256, 256, 3)
        self.attention_model = self._build_attention_model()

    def _build_attention_model(self) -> Model:
        """Build the attention model with proper feature sizes."""
        inputs = layers.Input(shape=self.input_shape)

        # Initial processing
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Store features at different scales
        features = []
        feature_sizes = [(256, 256), (128, 128), (64, 64)]
        current_filters = 64

        for size in feature_sizes:
            # Apply attention at current scale
            attention_module = AttentionModule(current_filters)
            attended, _ = attention_module(x)
            features.append(attended)

            # Downsample for next scale
            if size[0] > feature_sizes[-1][0]:
                x = layers.Conv2D(current_filters * 2, 3, strides=2, padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                current_filters *= 2

        # Upsample and concatenate features
        up_features = []
        for i, feature in enumerate(features):
            if i > 0:
                # Calculate upsampling size
                target_size = feature_sizes[0]
                feature = layers.Conv2DTranspose(
                    64, 3,
                    strides=2 ** (i),
                    padding='same'
                )(feature)
                feature = layers.BatchNormalization()(feature)
                feature = layers.Activation('relu')(feature)
            up_features.append(feature)

        # Concatenate features
        x = layers.Concatenate(axis=-1)(up_features)

        # Final attention
        final_attention = AttentionModule(256)
        output, attention_weights = final_attention(x)

        return Model(inputs, [output, attention_weights], name="attention_model")

    def enhance_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhance fingerprint using neural attention."""
        try:
            # Ensure image has correct shape and type
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            # Resize image to expected input size
            if image.shape[:2] != self.input_shape[:2]:
                image = tf.image.resize(image, self.input_shape[:2])

            # Normalize input
            image_tensor = tf.cast(image, tf.float32) / 255.0
            image_tensor = tf.expand_dims(image_tensor, 0)

            # Get attention features and maps
            try:
                features, attention_weights = self.attention_model(image_tensor)
            except Exception as e:
                self.logger.error(f"Error in attention model: {str(e)}")
                # Return unmodified fingerprint as fallback
                return fingerprint, np.ones_like(fingerprint)

            # Process attention weights
            attention_mask = tf.reduce_mean(attention_weights, axis=-1)
            attention_mask = tf.reshape(attention_mask, [-1, self.input_shape[0], self.input_shape[1], 1])
            attention_mask = tf.image.resize(attention_mask, fingerprint.shape[:2])
            attention_mask = attention_mask[0, ..., 0].numpy()

            # Normalize attention mask
            attention_mask = (attention_mask - attention_mask.min()) / (
                        attention_mask.max() - attention_mask.min() + 1e-8)

            # Apply attention to fingerprint
            enhanced_fingerprint = fingerprint * attention_mask

            return enhanced_fingerprint, attention_mask

        except Exception as e:
            self.logger.error(f"Error in enhance_fingerprint: {str(e)}")
            # Return original fingerprint in case of error
            return fingerprint, np.ones_like(fingerprint)

    def train(self, authentic_pairs, fake_pairs, epochs=1):
        """Train the attention model."""
        # Simple training implementation
        return {"loss": [0.0]}  # Mock training history

    def get_comparison_weights(self, fp1: np.ndarray, fp2: np.ndarray) -> np.ndarray:
        """Get attention weights for fingerprint comparison."""
        # Simple implementation returning uniform weights
        return np.ones_like(fp1)