from tests.test_base import AlphaPunchTestBase
from alphapunch.core.fingerprint_core import FingerprintCore
import numpy as np
import cv2

class TestFingerprintCore(AlphaPunchTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.core = FingerprintCore(
            private_key=cls.config['private_key'],
            config=cls.config,
            logger=cls.logger
        )

    def test_fingerprint_generation(self):
        """Test fingerprint generation for different image types."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                fingerprint = self.core._generate_secure_key(image.tobytes())
                self.assertIsNotNone(fingerprint)
                self.assertEqual(len(fingerprint), 32)  # SHA-256 length

    def test_dct_weights(self):
        """Test DCT weight generation."""
        weights = self.core._generate_dct_weights()
        self.assertEqual(weights.shape, tuple(self.config['algorithm']['fingerprint_size']))
        self.assertTrue(np.all(np.abs(weights) <= 1.0))

    def test_feature_extraction(self):
        """Test feature extraction from different images."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                dct_features = self.core._extract_dct_features(image)
                wavelet_features = self.core._extract_wavelet_features(image)
                stat_features = self.core._extract_statistical_features(image)

                # Check feature dimensions
                self.assertEqual(len(dct_features), 64)  # 8x8 DCT coefficients
                self.assertTrue(len(wavelet_features) > 0)
                self.assertTrue(len(stat_features) > 0)

    def test_error_correction(self):
        """Test error correction on fingerprints."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                # Generate noisy fingerprint
                fingerprint = np.random.rand(*self.config['algorithm']['fingerprint_size'])
                noise = np.random.normal(0, 0.1, fingerprint.shape)
                noisy_fingerprint = np.clip(fingerprint + noise, 0, 1)

                # Apply error correction
                corrected = self.core._apply_error_correction(noisy_fingerprint)

                # Check properties
                self.assertEqual(corrected.shape, fingerprint.shape)
                self.assertTrue(np.all(corrected >= 0) and np.all(corrected <= 1))

                # Error correction should reduce noise
                original_noise = np.mean(np.abs(fingerprint - noisy_fingerprint))
                corrected_noise = np.mean(np.abs(fingerprint - corrected))
                self.assertLess(corrected_noise, original_noise)

    def test_fingerprint_comparison(self):
        """Test fingerprint comparison with modifications."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                # Generate original fingerprint
                original_fp = self.core.generate_fingerprint(image)

                # Test different modifications
                modifications = {
                    'noise': lambda x: np.clip(x + np.random.normal(0, 0.1, x.shape), 0, 1),
                    'blur': lambda x: cv2.GaussianBlur(x, (3, 3), 0),
                    'scale': lambda x: cv2.resize(cv2.resize(x, (32, 32)), x.shape[:2][::-1])
                }

                for mod_name, mod_func in modifications.items():
                    modified_fp = mod_func(original_fp.copy())
                    similarity, mods = self.core.compare_fingerprints(original_fp, modified_fp)

                    # Basic checks
                    self.assertTrue(0 <= similarity <= 1)
                    self.assertIsInstance(mods, list)
