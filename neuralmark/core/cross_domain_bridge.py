import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class FeatureAlignment(layers.Layer):
    """Aligns features from different domains into a common space."""

    def __init__(self, alignment_dim=256):
        super(FeatureAlignment, self).__init__()
        self.alignment_dim = alignment_dim

        # Feature projection layers
        self.spatial_proj = layers.Dense(alignment_dim)
        self.freq_proj = layers.Dense(alignment_dim)
        self.wavelet_proj = layers.Dense(alignment_dim)

        # Feature normalization
        self.layer_norm = layers.LayerNormalization()

    def call(self, features):
        spatial, freq, wavelet = features['spatial'], features['frequency'], features['wavelet']

        # Project each domain to common dimension
        spatial_aligned = self.spatial_proj(self._flatten_features(spatial))
        freq_aligned = self.freq_proj(self._flatten_features(freq))
        wavelet_aligned = self.wavelet_proj(self._flatten_features(wavelet))

        # Normalize aligned features
        aligned_features = {
            'spatial': self.layer_norm(spatial_aligned),
            'frequency': self.layer_norm(freq_aligned),
            'wavelet': self.layer_norm(wavelet_aligned)
        }

        return aligned_features

    def _flatten_features(self, x):
        return tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])


class DomainCorrelation(layers.Layer):
    """Learns and maintains correlations between different domains."""

    def __init__(self, temperature=1.0):
        super(DomainCorrelation, self).__init__()
        self.temperature = temperature

        # Correlation learners
        self.key_transform = layers.Dense(128)
        self.query_transform = layers.Dense(128)

    def call(self, aligned_features):
        domains = ['spatial', 'frequency', 'wavelet']
        batch_size = tf.shape(aligned_features['spatial'])[0]

        correlation_matrix = tf.zeros([batch_size, len(domains), len(domains)])
        domain_weights = {}

        # Calculate correlation between each pair of domains
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if i <= j:  # Only calculate upper triangle
                    keys = self.key_transform(aligned_features[domain1])
                    queries = self.query_transform(aligned_features[domain2])

                    # Calculate correlation
                    correlation = tf.matmul(queries, keys, transpose_b=True)
                    correlation = correlation / self.temperature
                    correlation = tf.nn.softmax(correlation, axis=-1)

                    # Update correlation matrix
                    correlation_matrix = tf.tensor_scatter_nd_update(
                        correlation_matrix,
                        [[b, i, j] for b in range(batch_size)],
                        tf.reduce_mean(correlation, axis=[1, 2])
                    )

                    if i != j:  # Mirror for lower triangle
                        correlation_matrix = tf.tensor_scatter_nd_update(
                            correlation_matrix,
                            [[b, j, i] for b in range(batch_size)],
                            tf.reduce_mean(correlation, axis=[1, 2])
                        )

            # Calculate domain weights based on correlations
            domain_weights[domain1] = tf.nn.softmax(
                tf.reduce_mean(correlation_matrix[:, i, :], axis=1)
            )

        return correlation_matrix, domain_weights


class FeatureTransfer(layers.Layer):
    """Handles cross-domain feature transfer and enhancement."""

    def __init__(self, feature_dim=128):
        super(FeatureTransfer, self).__init__()
        self.feature_dim = feature_dim

        # Feature transformation layers
        self.transform = layers.Dense(feature_dim)
        self.gate = layers.Dense(feature_dim, activation='sigmoid')

        # Feature fusion
        self.fusion_layer = layers.Dense(feature_dim)

    def call(self, aligned_features, correlation_matrix, domain_weights):
        enhanced_features = {}

        for domain in aligned_features.keys():
            # Transform features
            domain_transform = self.transform(aligned_features[domain])

            # Calculate gate values
            gate_values = self.gate(
                tf.concat([domain_transform, tf.reduce_mean(correlation_matrix, axis=1)], axis=-1)
            )

            # Apply gating and weighting
            weighted_features = domain_transform * gate_values * \
                                tf.expand_dims(domain_weights[domain], axis=1)

            enhanced_features[domain] = weighted_features

        # Fuse enhanced features
        fused_features = tf.concat(list(enhanced_features.values()), axis=-1)
        fused_features = self.fusion_layer(fused_features)

        return enhanced_features, fused_features


class CrossDomainBridge(Model):
    """Main cross-domain bridge implementing feature alignment, correlation, and transfer."""

    def __init__(self, alignment_dim=256, feature_dim=128, temperature=1.0):
        super(CrossDomainBridge, self).__init__()

        # Initialize components
        self.feature_alignment = FeatureAlignment(alignment_dim)
        self.domain_correlation = DomainCorrelation(temperature)
        self.feature_transfer = FeatureTransfer(feature_dim)

        # Additional processing layers
        self.domain_fusion = layers.Dense(feature_dim)
        self.output_norm = layers.LayerNormalization()

    def call(self, domain_features):
        # Align features from different domains
        aligned_features = self.feature_alignment(domain_features)

        # Calculate domain correlations and weights
        correlation_matrix, domain_weights = self.domain_correlation(aligned_features)

        # Transfer and enhance features
        enhanced_features, fused_features = self.feature_transfer(
            aligned_features,
            correlation_matrix,
            domain_weights
        )

        # Final fusion and normalization
        output_features = self.domain_fusion(fused_features)
        output_features = self.output_norm(output_features)

        return output_features, {
            'correlation_matrix': correlation_matrix,
            'domain_weights': domain_weights,
            'enhanced_features': enhanced_features
        }

    def get_domain_relationships(self):
        """Returns the learned relationships between domains."""
        return {
            'correlation_matrix': self.domain_correlation.get_weights(),
            'domain_weights': self.feature_transfer.get_weights()
        }
    