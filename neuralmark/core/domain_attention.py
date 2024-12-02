import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import cv2
import pywt


class SpatialAttention(layers.Layer):
    """Multi-head self-attention for spatial domain."""

    def __init__(self, num_heads=8, key_dim=64):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )

        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.conv_position = layers.Conv2D(key_dim, 1, padding='same')

    def call(self, inputs):
        # Position-aware encoding
        position_features = self.conv_position(inputs)

        # Reshape for attention
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Flatten spatial dimensions
        queries = tf.reshape(position_features, [batch_size, -1, self.key_dim])
        keys = tf.reshape(inputs, [batch_size, -1, tf.shape(inputs)[-1]])

        # Apply attention
        attention_output = self.mha(
            query=queries,
            key=keys,
            value=keys
        )

        # Reshape back to spatial dimensions
        attention_output = tf.reshape(attention_output, [batch_size, height, width, -1])

        # Skip connection and normalization
        x1 = self.layernorm1(inputs + attention_output)

        return self.layernorm2(x1)


class FrequencyAttention(layers.Layer):
    """Attention mechanism for frequency domain features."""

    def __init__(self, num_bands=8):
        super(FrequencyAttention, self).__init__()
        self.num_bands = num_bands

        # Frequency band weighting
        self.band_weights = layers.Dense(num_bands, activation='softmax')
        self.freq_conv = layers.Conv2D(64, 3, padding='same')
        self.phase_attention = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Apply DCT
        dct_features = tf.signal.dct2d(tf.cast(inputs, tf.float32))

        # Split into frequency bands
        band_size = tf.shape(dct_features)[1] // self.num_bands
        bands = []

        for i in range(self.num_bands):
            start = i * band_size
            end = (i + 1) * band_size
            band = dct_features[:, start:end, :, :]
            bands.append(band)

        # Calculate band weights
        band_features = tf.stack([tf.reduce_mean(band, axis=[1, 2]) for band in bands], axis=1)
        weights = self.band_weights(band_features)

        # Apply weights to bands
        weighted_bands = [bands[i] * weights[:, i:i + 1, None, None] for i in range(self.num_bands)]
        weighted_dct = tf.concat(weighted_bands, axis=1)

        # Phase attention
        phase = tf.math.angle(tf.cast(inputs, tf.complex64))
        phase_weights = self.phase_attention(tf.reshape(phase, [tf.shape(phase)[0], -1]))
        phase_weights = tf.reshape(phase_weights, [tf.shape(phase)[0], 1, 1, 1])

        # Combine frequency and phase information
        freq_features = self.freq_conv(weighted_dct)
        freq_features = freq_features * phase_weights

        return tf.signal.idct2d(freq_features)


class WaveletAttention(layers.Layer):
    """Multi-scale wavelet coefficient attention."""

    def __init__(self, wavelet='db1', level=3):
        super(WaveletAttention, self).__init__()
        self.wavelet = wavelet
        self.level = level

        # Coefficient attention layers
        self.coeff_attention = layers.Dense(1, activation='sigmoid')
        self.detail_attention = layers.Dense(3, activation='softmax')  # For horizontal, vertical, diagonal
        self.scale_weights = layers.Dense(level, activation='softmax')

    def call(self, inputs):
        # Convert to numpy for wavelet transform
        inputs_np = inputs.numpy()
        batch_size = inputs_np.shape[0]
        attended_coeffs = []

        for b in range(batch_size):
            # Wavelet decomposition
            coeffs = pywt.wavedec2(inputs_np[b, ..., 0], self.wavelet, level=self.level)

            # Process approximation coefficient
            approx = coeffs[0]
            approx_weight = self.coeff_attention(tf.reshape(approx, [1, -1]))
            attended_approx = approx * approx_weight.numpy()

            # Process detail coefficients
            attended_details = []
            for level_coeffs in coeffs[1:]:
                # Get attention weights for each detail coefficient
                h, v, d = level_coeffs
                details = np.stack([h, v, d])
                detail_weights = self.detail_attention(tf.reshape(details, [1, -1]))

                # Apply attention weights
                attended_h = h * detail_weights[0].numpy()
                attended_v = v * detail_weights[1].numpy()
                attended_d = d * detail_weights[2].numpy()

                attended_details.append((attended_h, attended_v, attended_d))

            # Reconstruct with attended coefficients
            attended_coeffs.append(
                pywt.waverec2([attended_approx] + attended_details, self.wavelet)
            )

        # Convert back to tensor
        return tf.convert_to_tensor(np.stack(attended_coeffs, axis=0))


class DomainSpecificAttention(Model):
    """Combines spatial, frequency, and wavelet attention mechanisms."""

    def __init__(self, num_heads=8, freq_bands=8, wavelet_level=3):
        super(DomainSpecificAttention, self).__init__()

        # Initialize attention mechanisms
        self.spatial_attention = SpatialAttention(num_heads=num_heads)
        self.frequency_attention = FrequencyAttention(num_bands=freq_bands)
        self.wavelet_attention = WaveletAttention(level=wavelet_level)

        # Feature combination layers
        self.feature_conv = layers.Conv2D(64, 3, padding='same')
        self.feature_norm = layers.LayerNormalization()
        self.output_conv = layers.Conv2D(32, 1, padding='same')

    def call(self, inputs):
        # Apply domain-specific attention
        spatial_features = self.spatial_attention(inputs)
        frequency_features = self.frequency_attention(inputs)
        wavelet_features = self.wavelet_attention(inputs)

        # Combine features
        combined = tf.concat([
            spatial_features,
            frequency_features,
            wavelet_features
        ], axis=-1)

        # Process combined features
        x = self.feature_conv(combined)
        x = self.feature_norm(x)
        outputs = self.output_conv(x)

        return outputs, {
            'spatial': spatial_features,
            'frequency': frequency_features,
            'wavelet': wavelet_features
        }