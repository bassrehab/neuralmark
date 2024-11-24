from tests.test_base import AlphaPunchTestBase
from alphapunch.core.amdf import AdaptiveMultiDomainFingerprinting
import tensorflow as tf
import numpy as np
import cv2


class TestAMDF(AlphaPunchTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.amdf = AdaptiveMultiDomainFingerprinting(
            config=cls.config,
            logger=cls.logger
        )

    def test_feature_extractor(self):
        """Test VGG-based feature extraction."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                # Prepare input
                preprocessed = tf.image.resize(image, (224, 224))
                preprocessed = tf.expand_dims(preprocessed, 0)

                # Extract features
                features = self.amdf.feature_extractor(preprocessed)

                # Check output shape
                self.assertEqual(len(features.shape), 4)  # [batch, height, width, channels]
                self.assertEqual(features.shape[0], 1)  # Batch size

    def test_fingerprint_generator(self):
        """Test fingerprint generation network."""
        # Create dummy features
        dummy_features = np.random.rand(1, 28, 28, 256).astype(np.float32)

        # Generate fingerprint
        fingerprint = self.amdf.fingerprint_generator(dummy_features)

        # Check output
        self.assertEqual(fingerprint.shape[1:3], tuple(self.config['algorithm']['fingerprint_size']))
        self.assertTrue(np.all(fingerprint >= 0) and np.all(fingerprint <= 1))

    def test_verifier(self):
        """Test verification network."""
        fp_size = self.config['algorithm']['fingerprint_size']

        # Create test fingerprints
        fp1 = np.random.rand(1, fp_size[0], fp_size[1], 1).astype(np.float32)
        fp2 = np.random.rand(1, fp_size[0], fp_size[1], 1).astype(np.float32)

        # Get similarity score
        similarity = self.amdf.verifier([fp1, fp2])

        # Check output
        self.assertTrue(0 <= float(similarity) <= 1)

    def test_multi_domain_features(self):
        """Test extraction of features from multiple domains."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                # Test each domain
                features = []

                # VGG features
                preprocessed = tf.image.resize(image, (224, 224))
                preprocessed = tf.expand_dims(preprocessed, 0)
                vgg_features = self.amdf.feature_extractor(preprocessed)
                features.append(vgg_features)

                # Wavelet features
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                wavelet_features = self.amdf._extract_wavelet_features(gray)
                features.append(wavelet_features)

                # Check feature properties
                for feature in features:
                    self.assertIsNotNone(feature)
                    self.assertGreater(np.sum(np.abs(feature)), 0)

    def test_compare_wavelet(self):
        """Test wavelet-based fingerprint comparison."""
        for name, image in self.test_images.items():
            with self.subTest(image_type=name):
                # Generate two similar fingerprints
                fp1 = self.amdf.generate_fingerprint(image)

                # Add small modification
                modified = cv2.GaussianBlur(image, (3, 3), 0)
                fp2 = self.amdf.generate_fingerprint(modified)

                # Compare wavelets
                similarity = self.amdf._compare_wavelets(fp1, fp2)

                # Check similarity properties
                self.assertTrue(0 <= similarity <= 1)
                self.assertGreater(similarity, 0.5)  # Should be similar

    def test_modification_detection(self):
        """Test detection of various modifications."""
        # Generate base fingerprint
        base_image = self.test_images['pattern']
        base_fp = self.amdf.generate_fingerprint(base_image)

        # Test different modifications
        modifications = {
            'jpeg': lambda x: cv2.imdecode(
                cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, 50])[1],
                cv2.IMREAD_COLOR
            ),
            'geometric': lambda x: cv2.resize(cv2.resize(x, (128, 128)), (256, 256)),
            'intensity': lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=20)
        }

        for mod_name, mod_func in modifications.items():
            modified = mod_func(base_image)
            mod_fp = self.amdf.generate_fingerprint(modified)

            # Detect modifications
            similarity, detected_mods = self.amdf.compare_fingerprints(base_fp, mod_fp)

            # Check if modification was detected
            self.assertGreater(len(detected_mods), 0)

    def test_training(self):
        """Test verifier training."""
        # Create training data
        authentic_pairs = []
        fake_pairs = []

        for name, image in self.test_images.items():
            # Create authentic pair
            fp = self.amdf.generate_fingerprint(image)
            modified = cv2.GaussianBlur(image, (3, 3), 0)
            authentic_pairs.append((image, modified))

            # Create fake pair
            fake_image = np.random.randint(0, 255, image.shape, dtype=np.uint8)
            fake_pairs.append((image, fake_image))

        # Train verifier
        self.amdf.train_verifier(authentic_pairs, fake_pairs)

        # Test after training
        test_image = self.test_images['pattern']
        fp1 = self.amdf.generate_fingerprint(test_image)
        modified = cv2.GaussianBlur(test_image, (3, 3), 0)
        fp2 = self.amdf.generate_fingerprint(modified)

        similarity, _ = self.amdf.compare_fingerprints(fp1, fp2)
        self.assertGreater(similarity, 0.5)


if __name__ == '__main__':
    unittest.main()