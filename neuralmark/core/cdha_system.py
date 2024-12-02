import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

from .domain_attention import DomainSpecificAttention
from .cross_domain_bridge import CrossDomainBridge
from .hierarchical_fusion import HierarchicalFusion
from .adaptive_weight_generator import AdaptiveWeightGenerator


class CrossDomainHierarchicalAttention(Model):
    """Complete CDHA system integrating all components."""

    def __init__(self, config: dict):
        super(CrossDomainHierarchicalAttention, self).__init__()

        # Extract configuration parameters
        self.feature_dim = config['algorithm']['neural_attention']['feature_channels']
        self.num_heads = config['algorithm']['neural_attention'].get('num_heads', 8)
        self.num_attack_types = config['algorithm']['neural_attention'].get('num_attack_types', 5)

        # Initialize components
        self.domain_attention = DomainSpecificAttention(
            num_heads=self.num_heads,
            freq_bands=8,
            wavelet_level=3
        )

        self.cross_domain_bridge = CrossDomainBridge(
            alignment_dim=self.feature_dim,
            feature_dim=self.feature_dim,
            temperature=1.0
        )

        self.hierarchical_fusion = HierarchicalFusion(
            low_dim=self.feature_dim // 4,
            mid_dim=self.feature_dim // 2,
            high_dim=self.feature_dim
        )

        self.weight_generator = AdaptiveWeightGenerator(
            feature_dim=self.feature_dim,
            num_attack_types=self.num_attack_types
        )

        # Additional processing layers
        self.feature_refinement = layers.Dense(self.feature_dim)
        self.output_conv = layers.Conv2D(1, 1, padding='same')

    def call(self, inputs, training=False, feedback=None):
        # Generate adaptive weights
        adaptive_weights, weight_info = self.weight_generator(inputs, feedback)

        # Apply domain-specific attention
        domain_features, domain_info = self.domain_attention(inputs)

        # Bridge different domains
        bridged_features, bridge_info = self.cross_domain_bridge(domain_info)

        # Apply hierarchical fusion
        fused_features, fusion_info = self.hierarchical_fusion(inputs, bridged_features)

        # Apply adaptive weights
        weighted_features = fused_features * tf.reshape(adaptive_weights, [-1, 1, 1, 1])

        # Final processing
        refined = self.feature_refinement(
            tf.reshape(weighted_features, [tf.shape(inputs)[0], -1])
        )
        outputs = self.output_conv(
            tf.reshape(refined, [tf.shape(inputs)[0], 1, 1, -1])
        )

        return outputs, {
            'domain_info': domain_info,
            'bridge_info': bridge_info,
            'fusion_info': fusion_info,
            'weight_info': weight_info
        }

    def get_attention_maps(self, inputs):
        """Generate attention maps for visualization."""
        _, domain_info = self.domain_attention(inputs)
        _, bridge_info = self.cross_domain_bridge(domain_info)

        return {
            'spatial_attention': domain_info['spatial'],
            'frequency_attention': domain_info['frequency'],
            'wavelet_attention': domain_info['wavelet'],
            'domain_correlation': bridge_info['correlation_matrix']
        }

    def analyze_vulnerability(self, inputs):
        """Perform vulnerability analysis of the input."""
        return self.weight_generator.get_vulnerability_assessment(inputs)


class CDHAFingerprinter:
    """Main interface for using CDHA in fingerprinting applications."""

    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger

        # Initialize CDHA model
        self.cdha = CrossDomainHierarchicalAttention(config)

        # Set up image preprocessing
        self.target_size = tuple(config['algorithm']['input_shape'][:2])

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Generate fingerprint using CDHA."""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Generate fingerprint
            fingerprint, info = self.cdha(processed_image)

            # Postprocess fingerprint
            final_fingerprint = self._postprocess_fingerprint(fingerprint)

            if self.logger:
                self.logger.debug("Fingerprint generated successfully")

            return final_fingerprint

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating fingerprint: {str(e)}")
            raise

    def verify_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> dict:
        """Verify fingerprint using CDHA."""
        try:
            # Analyze vulnerability
            vulnerability = self.cdha.analyze_vulnerability(
                self._preprocess_image(image)
            )

            # Generate attention maps
            attention_maps = self.cdha.get_attention_maps(
                self._preprocess_image(image)
            )

            return {
                'vulnerability': vulnerability,
                'attention_maps': attention_maps
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error verifying fingerprint: {str(e)}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """Preprocess image for CDHA."""
        # Resize image
        resized = tf.image.resize(image, self.target_size)

        # Normalize
        normalized = resized / 255.0

        # Add batch dimension
        batched = tf.expand_dims(normalized, 0)

        return batched

    def _postprocess_fingerprint(self, fingerprint: tf.Tensor) -> np.ndarray:
        """Postprocess CDHA output into final fingerprint."""
        # Remove batch dimension
        squeezed = tf.squeeze(fingerprint)

        # Normalize to [0, 1] range
        normalized = tf.keras.layers.Lambda(
            lambda x: (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
        )(squeezed)

        return normalized.numpy()


def create_cdha_fingerprinter(config: dict, logger=None) -> CDHAFingerprinter:
    """Factory function to create CDHA fingerprinter."""
    return CDHAFingerprinter(config, logger)