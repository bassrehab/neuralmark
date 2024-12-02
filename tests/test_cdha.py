import pytest
import numpy as np
import cv2
import tensorflow as tf

from neuralmark.core.cdha_system import CrossDomainHierarchicalAttention, CDHAFingerprinter


class TestCDHA:
    @pytest.fixture
    def cdha_config(self, test_config):
        """Provide CDHA-specific configuration."""
        test_config['algorithm_selection']['type'] = 'cdha'
        test_config['algorithm']['cdha'] = {
            'feature_weights': {
                'spatial': 0.4,
                'frequency': 0.3,
                'wavelet': 0.3
            },
            'attention': {
                'num_heads': 8,
                'key_dim': 64,
                'num_attack_types': 5
            },
            'hierarchical': {
                'low_dim': 64,
                'mid_dim': 128,
                'high_dim': 256
            }
        }
        return test_config

    @pytest.fixture
    def cdha(self, cdha_config, test_logger):
        return CrossDomainHierarchicalAttention(cdha_config)

    @pytest.fixture
    def fingerprinter(self, cdha_config, test_logger):
        return CDHAFingerprinter(cdha_config, test_logger)

    def test_initialization(self, cdha):
        """Test CDHA initialization."""
        assert cdha is not None
        assert hasattr(cdha, 'domain_attention')
        assert hasattr(cdha, 'cross_domain_bridge')
        assert hasattr(cdha, 'hierarchical_fusion')
        assert hasattr(cdha, 'weight_generator')

    def test_domain_attention(self, cdha, test_image):
        """Test domain-specific attention."""
        # Convert image to tensor
        image_tensor = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension

        # Get domain features
        domain_features, domain_info = cdha.domain_attention(image_tensor)

        assert isinstance(domain_features, tf.Tensor)
        assert isinstance(domain_info, dict)
        assert 'spatial' in domain_info
        assert 'frequency' in domain_info
        assert 'wavelet' in domain_info

    def test_cross_domain_bridge(self, cdha, test_image):
        """Test cross-domain feature bridging."""
        image_tensor = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        # Get domain features
        domain_features, domain_info = cdha.domain_attention(image_tensor)

        # Test bridge
        bridged_features, bridge_info = cdha.cross_domain_bridge(domain_info)

        assert isinstance(bridged_features, tf.Tensor)
        assert isinstance(bridge_info, dict)
        assert 'correlation_matrix' in bridge_info
        assert 'domain_weights' in bridge_info

    def test_hierarchical_fusion(self, cdha, test_image):
        """Test hierarchical feature fusion."""
        image_tensor = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        # Get features through pipeline
        domain_features, domain_info = cdha.domain_attention(image_tensor)
        bridged_features, _ = cdha.cross_domain_bridge(domain_info)

        # Test fusion
        fused_features, fusion_info = cdha.hierarchical_fusion(image_tensor, bridged_features)

        assert isinstance(fused_features, tf.Tensor)
        assert isinstance(fusion_info, dict)
        assert all(k in fusion_info for k in ['low_level', 'mid_level', 'high_level'])

    def test_weight_generator(self, cdha, test_image):
        """Test adaptive weight generation."""
        image_tensor = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        # Generate weights
        weights, weight_info = cdha.weight_generator(image_tensor)

        assert isinstance(weights, tf.Tensor)
        assert isinstance(weight_info, dict)
        assert 'content_features' in weight_info
        assert 'attack_features' in weight_info

    def test_fingerprint_generation(self, fingerprinter, test_image):
        """Test complete fingerprint generation."""
        fingerprint = fingerprinter.generate_fingerprint(test_image)

        assert isinstance(fingerprint, np.ndarray)
        assert fingerprint.shape == tuple(fingerprinter.config['algorithm']['fingerprint_size'])
        assert fingerprint.dtype == np.float32
        assert np.all((fingerprint >= 0) & (fingerprint <= 1))

    def test_fingerprint_verification(self, fingerprinter, test_image):
        """Test fingerprint verification."""
        # Generate and verify fingerprint
        fingerprint = fingerprinter.generate_fingerprint(test_image)
        verification_results = fingerprinter.verify_fingerprint(test_image, fingerprint)

        assert isinstance(verification_results, dict)
        assert 'vulnerability' in verification_results
        assert 'attention_maps' in verification_results

    def test_attention_visualization(self, fingerprinter, test_image):
        """Test attention map generation."""
        attention_maps = fingerprinter.cdha.get_attention_maps(
            tf.convert_to_tensor(test_image, dtype=tf.float32)
        )

        assert isinstance(attention_maps, dict)
        assert 'spatial_attention' in attention_maps
        assert 'frequency_attention' in attention_maps
        assert 'wavelet_attention' in attention_maps
        assert 'domain_correlation' in attention_maps

    def test_robustness_analysis(self, fingerprinter, test_image):
        """Test robustness against various modifications."""
        # Generate fingerprint
        original_fingerprint = fingerprinter.generate_fingerprint(test_image)

        modifications = {
            'blur': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            'noise': lambda img: np.clip(img + np.random.normal(0, 10, img.shape), 0, 255).astype(np.uint8),
            'rotate': lambda img: cv2.warpAffine(img,
                                                 cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 5, 1),
                                                 (img.shape[1], img.shape[0]))
        }

        for mod_name, mod_func in modifications.items():
            # Apply modification
            modified_image = mod_func(test_image)

            # Verify fingerprint
            results = fingerprinter.verify_fingerprint(modified_image, original_fingerprint)

            # Check vulnerability assessment
            assert 'vulnerability' in results
            assert 0 <= results['vulnerability']['vulnerability_score'] <= 1

            # Check attention maps
            assert 'attention_maps' in results
            assert all(k in results['attention_maps'] for k in [
                'spatial_attention', 'frequency_attention', 'wavelet_attention'
            ])

    @pytest.mark.parametrize("attack_type", ['compression', 'geometric', 'noise'])
    def test_attack_detection(self, fingerprinter, test_image, attack_type):
        """Test specific attack type detection."""
        fingerprint = fingerprinter.generate_fingerprint(test_image)
        results = fingerprinter.cdha.analyze_vulnerability(
            tf.convert_to_tensor(test_image, dtype=tf.float32)
        )

        assert 'vulnerability_score' in results
        assert 'attack_probabilities' in results
        assert 'robustness_score' in results
