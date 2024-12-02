import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from neuralmark.core.neural_attention import (
    SpatialAttention,
    NeuralAttentionEnhancer
)


class TestNeuralAttention:
    @pytest.fixture
    def attention_config(self, test_config):
        """Provide neural attention specific configuration."""
        test_config['algorithm']['neural_attention'] = {
            'enabled': True,
            'attention_layers': 3,
            'base_filters': 64,
            'feature_channels': 256
        }
        return test_config

    @pytest.fixture
    def neural_attention(self, attention_config, test_logger):
        return NeuralAttentionEnhancer(attention_config, test_logger)

    def test_initialization(self, neural_attention):
        """Test neural attention initialization."""
        assert neural_attention is not None
        assert hasattr(neural_attention, 'attention_model')
        assert isinstance(neural_attention.attention_model.layers[0], layers.Input)

    def test_spatial_attention_layer(self):
        """Test spatial attention layer."""
        spatial_attention = SpatialAttention(filters=32)

        # Create test input
        test_input = tf.random.normal([1, 64, 64, 32])
        output = spatial_attention(test_input)

        # Test output shape and type
        assert isinstance(output, tuple)
        assert isinstance(output[0], tf.Tensor)
        assert output[0].shape.as_list() == [1, 64, 64, 32]

    def test_enhance_fingerprint(self, neural_attention, test_image, test_fingerprint):
        """Test fingerprint enhancement."""
        # Prepare inputs
        image = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)  # Add batch dimension

        fingerprint = tf.convert_to_tensor(test_fingerprint, dtype=tf.float32)

        # Enhance fingerprint
        enhanced_fp, attention_mask = neural_attention.enhance_fingerprint(
            image, fingerprint
        )

        # Verify outputs
        assert isinstance(enhanced_fp, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert enhanced_fp.shape == test_fingerprint.shape
        assert attention_mask.shape == test_fingerprint.shape
        assert np.all(np.isfinite(enhanced_fp))
        assert np.all(np.isfinite(attention_mask))
        assert np.all((attention_mask >= 0) & (attention_mask <= 1))

    def test_attention_model_output(self, neural_attention, test_image):
        """Test attention model output."""
        # Prepare input
        image = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)

        # Get model outputs
        features, attention = neural_attention.attention_model(image, training=False)

        assert isinstance(features, tf.Tensor)
        assert isinstance(attention, tf.Tensor)
        assert len(features.shape) == 4  # [batch, height, width, channels]
        assert len(attention.shape) >= 2  # Attention weights

    def test_attention_visualization(self, neural_attention, test_image):
        """Test attention visualization capability."""
        image = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)

        _, attention = neural_attention.attention_model(image, training=False)

        # Convert attention to visualization format
        attention_map = tf.reduce_mean(attention, axis=-1)
        attention_map = tf.reshape(attention_map, [-1] + test_image.shape[:2])

        assert attention_map.shape[1:] == test_image.shape[:2]
        assert np.all(np.isfinite(attention_map))

    def test_training_mode(self, neural_attention, test_image):
        """Test model behavior in training mode."""
        image = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)

        # Test with training=True
        features_train, attention_train = neural_attention.attention_model(
            image, training=True
        )

        # Test with training=False
        features_test, attention_test = neural_attention.attention_model(
            image, training=False
        )

        # Outputs should have same shapes but potentially different values
        assert features_train.shape == features_test.shape
        assert attention_train.shape == attention_test.shape

    @pytest.mark.parametrize("image_size", [(128, 128), (256, 256)])
    def test_different_input_sizes(self, neural_attention, image_size):
        """Test handling of different input sizes."""
        # Create test image
        test_input = np.random.rand(1, image_size[0], image_size[1], 3)
        test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)

        # Test model with different input sizes
        features, attention = neural_attention.attention_model(test_input, training=False)

        assert features.shape[1:3] == test_input.shape[1:3]  # Spatial dimensions should match

    def test_feature_enhancement(self, neural_attention, test_image):
        """Test feature enhancement capabilities."""
        # Original features
        image = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)

        features, _ = neural_attention.attention_model(image, training=False)

        # Create synthetic fingerprint
        fingerprint = np.random.rand(*neural_attention.input_shape[:2])

        # Enhanced features
        enhanced_fp, _ = neural_attention.enhance_fingerprint(image, fingerprint)

        assert enhanced_fp.shape == fingerprint.shape
        assert np.any(enhanced_fp != fingerprint)  # Should be modified

    def test_compare_attention_maps(self, neural_attention, test_image):
        """Test comparison of attention maps."""
        image1 = tf.convert_to_tensor(test_image, dtype=tf.float32)
        image1 = tf.expand_dims(image1, 0)

        # Create slightly modified image
        image2 = test_image + np.random.normal(0, 10, test_image.shape)
        image2 = np.clip(image2, 0, 255).astype(np.uint8)
        image2 = tf.convert_to_tensor(image2, dtype=tf.float32)
        image2 = tf.expand_dims(image2, 0)

        # Get attention maps
        _, attention1 = neural_attention.attention_model(image1, training=False)
        _, attention2 = neural_attention.attention_model(image2, training=False)

        # Compare attention maps
        attention_diff = tf.reduce_mean(tf.abs(attention1 - attention2))

        assert attention_diff > 0  # Should be some difference
        assert attention_diff < 1  # But not completely different

    def test_error_handling(self, neural_attention):
        """Test error handling for invalid inputs."""
        # Test with invalid image shape
        invalid_image = np.random.rand(64, 64)  # Missing channel dimension
        invalid_fingerprint = np.random.rand(32, 32)  # Wrong size

        with pytest.raises(ValueError):
            neural_attention.enhance_fingerprint(invalid_image, invalid_fingerprint)

    def test_get_comparison_weights(self, neural_attention, test_fingerprint):
        """Test comparison weights generation."""
        fp1 = test_fingerprint
        fp2 = test_fingerprint + np.random.normal(0, 0.1, test_fingerprint.shape)

        weights = neural_attention.get_comparison_weights(fp1, fp2)

        assert weights.shape == fp1.shape
        assert np.all((weights >= 0) & (weights <= 1))
