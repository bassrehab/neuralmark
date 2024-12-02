import pytest
import numpy as np
import cv2

from neuralmark.core.amdf import AdaptiveMultiDomainFingerprinting


class TestAMDF:
    @pytest.fixture
    def amdf(self, test_config, test_logger):
        return AdaptiveMultiDomainFingerprinting(test_config, test_logger)

    def test_initialization(self, amdf):
        """Test AMDF initialization."""
        assert amdf is not None
        assert hasattr(amdf, 'feature_extractor')
        assert hasattr(amdf, 'fingerprint_generator')
        assert hasattr(amdf, 'verifier')

    def test_generate_fingerprint(self, amdf, test_image):
        """Test fingerprint generation."""
        fingerprint = amdf.generate_fingerprint(test_image)

        # Check fingerprint properties
        assert fingerprint.shape == tuple(amdf.config['algorithm']['fingerprint_size'])
        assert fingerprint.dtype == np.float32
        assert np.all((fingerprint >= 0) & (fingerprint <= 1))

    def test_embed_fingerprint(self, amdf, test_image, test_fingerprint):
        """Test fingerprint embedding."""
        embedded = amdf.embed_fingerprint(test_image, test_fingerprint)

        # Check embedded image properties
        assert embedded.shape == test_image.shape
        assert embedded.dtype == np.uint8
        assert np.any(embedded != test_image)  # Should be different from original

    def test_extract_fingerprint(self, amdf, test_image):
        """Test fingerprint extraction."""
        # Generate and embed fingerprint
        fingerprint = amdf.generate_fingerprint(test_image)
        embedded = amdf.embed_fingerprint(test_image, fingerprint)

        # Extract fingerprint
        extracted = amdf.extract_fingerprint(embedded)

        # Check extracted fingerprint properties
        assert extracted.shape == fingerprint.shape
        assert extracted.dtype == fingerprint.dtype
        assert np.corrcoef(extracted.flatten(), fingerprint.flatten())[0, 1] > 0.5

    def test_compare_fingerprints(self, amdf, test_fingerprint):
        """Test fingerprint comparison."""
        # Create slightly modified fingerprint
        modified = test_fingerprint + np.random.normal(0, 0.1, test_fingerprint.shape)
        modified = np.clip(modified, 0, 1)

        # Compare fingerprints
        similarity, modifications = amdf.compare_fingerprints(test_fingerprint, modified)

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        assert isinstance(modifications, list)

    def test_verify_fingerprint(self, amdf, test_image):
        """Test fingerprint verification."""
        # Generate and embed fingerprint
        fingerprint = amdf.generate_fingerprint(test_image)
        embedded = amdf.embed_fingerprint(test_image, fingerprint)

        # Verify fingerprint
        is_authentic, similarity, modifications = amdf.verify_fingerprint(
            embedded, fingerprint
        )

        assert isinstance(is_authentic, bool)
        assert isinstance(similarity, float)
        assert isinstance(modifications, list)
        assert is_authentic  # Should verify successfully

    def test_robustness_to_modifications(self, amdf, test_image):
        """Test robustness against various modifications."""
        # Generate and embed fingerprint
        fingerprint = amdf.generate_fingerprint(test_image)
        embedded = amdf.embed_fingerprint(test_image, fingerprint)

        modifications = {
            'blur': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            'noise': lambda img: np.clip(img + np.random.normal(0, 10, img.shape), 0, 255).astype(np.uint8),
            'compress': lambda img: cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(),
            'crop': lambda img: img[10:-10, 10:-10]
        }

        for mod_name, mod_func in modifications.items():
            # Apply modification
            try:
                modified = mod_func(embedded)
                if isinstance(modified, bytes):
                    modified = cv2.imdecode(np.frombuffer(modified, np.uint8), cv2.IMREAD_COLOR)

                # Verify fingerprint
                is_authentic, similarity, mods = amdf.verify_fingerprint(
                    modified, fingerprint
                )

                # Should still detect some similarity
                assert similarity > 0.3

            except Exception as e:
                pytest.fail(f"Error testing {mod_name} modification: {str(e)}")

    def test_different_image_rejection(self, amdf, test_image):
        """Test rejection of different images."""
        # Generate fingerprint for first image
        fingerprint1 = amdf.generate_fingerprint(test_image)

        # Create different image
        different_image = np.zeros_like(test_image)
        cv2.rectangle(different_image, (50, 50), (200, 200), (0, 255, 0), -1)

        # Verify fingerprint with different image
        is_authentic, similarity, _ = amdf.verify_fingerprint(
            different_image, fingerprint1
        )

        assert not is_authentic
        assert similarity < 0.5  # Should have low similarity

    @pytest.mark.parametrize("image_size", [(128, 128), (512, 512)])
    def test_different_image_sizes(self, amdf, image_size):
        """Test handling of different image sizes."""
        # Create test image of different size
        image = np.zeros((*image_size, 3), dtype=np.uint8)
        cv2.circle(image, (image_size[0] // 2, image_size[1] // 2),
                   min(image_size) // 4, (255, 0, 0), -1)

        # Test fingerprint generation
        fingerprint = amdf.generate_fingerprint(image)
        assert fingerprint.shape == tuple(amdf.config['algorithm']['fingerprint_size'])

        # Test embedding
        embedded = amdf.embed_fingerprint(image, fingerprint)
        assert embedded.shape == image.shape
