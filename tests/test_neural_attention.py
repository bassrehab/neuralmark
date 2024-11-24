from tests.test_base import AlphaPunchTestBase
from alphapunch.core.neural_attention import NeuralAttentionEnhancer
import tensorflow as tf
import numpy as np


class TestNeuralAttention(AlphaPunchTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.attention = NeuralAttentionEnhancer(cls.config, cls.logger)

    def test_attention_module(self):
        """Test attention module functionality."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                # Prepare input
                image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32) / 255.0

                # Get attention features and maps
                features, attention_maps = self.attention.attention_model(image_tensor)

                # Check shapes and values
                self.assertEqual(len(attention_maps), 3)  # Number of attention layers
                self.assertTrue(tf.reduce_all(attention_maps[0] >= 0))
                self.assertTrue(tf.reduce_all(tf.reduce_sum(attention_maps[0], axis=-1) <= 1.0001))

    def test_fingerprint_enhancement(self):
        """Test fingerprint enhancement with attention."""
        fingerprint = np.random.rand(*self.config['algorithm']['fingerprint_size'])

        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                enhanced_fp, attention_mask = self.attention.enhance_fingerprint(image, fingerprint)

                # Check shapes
                self.assertEqual(enhanced_fp.shape, fingerprint.shape)
                self.assertEqual(attention_mask.shape[:2], fingerprint.shape[:2])

                # Check value ranges
                self.assertTrue(np.all(enhanced_fp >= 0) and np.all(enhanced_fp <= 1))
                self.assertTrue(np.all(attention_mask >= 0) and np.all(attention_mask <= 1))

    def test_attention_adaptation(self):
        """Test if attention adapts to image content."""
        # Test with blank and pattern images
        blank_fp = np.random.rand(*self.config['algorithm']['fingerprint_size'])
        _, blank_attention = self.attention.enhance_fingerprint(
            self.test_images['blank'], blank_fp
        )

        _, pattern_attention = self.attention.enhance_fingerprint(
            self.test_images['pattern'], blank_fp
        )

        # Pattern image should have more varied attention
        blank_std = np.std(blank_attention)
        pattern_std = np.std(pattern_attention)
        self.assertGreater(pattern_std, blank_std)

    def test_training(self):
        """Test attention model training."""
        # Create small training set
        train_images = [img for img in self.test_images.values()]
        train_pairs = [(img, img) for img in train_images]  # Same image pairs
        fake_pairs = [(img1, img2) for img1, img2 in zip(train_images[:-1], train_images[1:])]

        # Train for one epoch
        history = self.attention.train(train_pairs, fake_pairs, epochs=1)

        # Check training outputs
        self.assertIn('loss', history.history)
        self.assertGreater(len(history.history['loss']), 0)
