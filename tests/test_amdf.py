import unittest
import numpy as np
from neuralmark.core.amdf import AdaptiveMultiDomainFingerprinting


class TestAdaptiveMultiDomainFingerprinting(unittest.TestCase):
    def setUp(self):
        """Set up an instance of the AdaptiveMultiDomainFingerprinting class."""
        self.amdf = AdaptiveMultiDomainFingerprinting(config={'key': 'value'})
        self.sample_image = np.random.rand(256, 256, 3) * 255  # Simulate a random image

    def test_initialization(self):
        """Test initialization of the class."""
        self.assertIsNotNone(self.amdf)
        self.assertEqual(self.amdf.config['key'], 'value')

    def test_feature_extraction(self):
        """Test feature extraction from an image."""
        features = self.amdf.feature_extractor(self.sample_image)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)

    def test_fingerprint_generation(self):
        """Test fingerprint generation functionality."""
        fingerprint = self.amdf._build_fingerprint_generator()
        self.assertIsNotNone(fingerprint)

    def test_verifier_training(self):
        """Test training the verifier model."""
        training_data = [np.random.rand(256, 256, 3) for _ in range(10)]  # Simulate training data
        labels = np.random.randint(0, 2, 10)  # Binary labels
        result = self.amdf.train_verifier(training_data, labels)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
