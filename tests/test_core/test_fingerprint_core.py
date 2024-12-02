import pytest
import numpy as np
import cv2

from neuralmark.core.fingerprint_core import FingerprintCore


class TestFingerprintCore:
    @pytest.fixture
    def fingerprint_core(self, test_config, test_logger):
        return FingerprintCore(
            private_key="test-key-2024",
            config=test_config,
            logger=test_logger
        )

    def test_initialization(self, fingerprint_core):
        """Test core initialization."""
        assert fingerprint_core.private_key is not None
        assert fingerprint_core.fingerprint_size == tuple(fingerprint_core.config['algorithm']['fingerprint_size'])
        assert fingerprint_core.dct_weights.shape == fingerprint_core.fingerprint_size

    def test_secure_key_generation(self, fingerprint_core):
        """Test secure key generation."""
        key1 = fingerprint_core._generate_secure_key("test-key-1")
        key2 = fingerprint_core._generate_secure_key("test-key-2")

        assert isinstance(key1, bytes)
        assert len(key1) == 32  # SHA-256 output
        assert key1 != key2  # Different keys should produce different results

    def test_dct_weights_generation(self, fingerprint_core):
        """Test DCT weights generation."""
        weights = fingerprint_core._generate_dct_weights()

        assert isinstance(weights, np.ndarray)
        assert weights.shape == fingerprint_core.fingerprint_size
        assert np.all(np.abs(weights) <= 1)  # Weights should be normalized

    def test_extract_features(self, fingerprint_core, test_image):
        """Test feature extraction."""
        # Extract features at different scales
        features = []
        for scale in fingerprint_core.scales:
            scale_features = fingerprint_core._extract_features_at_scale(test_image, scale)
            features.append(scale_features)

        assert len(features) == len(fingerprint_core.scales)
        assert all(isinstance(f, np.ndarray) for f in features)

    def test_dct_feature_extraction(self, fingerprint_core, test_image):
        """Test DCT feature extraction."""
        # Convert to grayscale if needed
        if len(test_image.shape) == 3:
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = test_image

        dct_features = fingerprint_core._extract_dct_features(gray)

        assert isinstance(dct_features, np.ndarray)
        assert len(dct_features.shape) == 1
        assert np.all(np.isfinite(dct_features))

    def test_wavelet_feature_extraction(self, fingerprint_core, test_image):
        """Test wavelet feature extraction."""
        if len(test_image.shape) == 3:
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = test_image

        wavelet_features = fingerprint_core._extract_wavelet_features(gray)

        assert isinstance(wavelet_features, np.ndarray)
        assert len(wavelet_features.shape) == 1
        assert np.all(np.isfinite(wavelet_features))

    def test_statistical_feature_extraction(self, fingerprint_core, test_image):
        """Test statistical feature extraction."""
        if len(test_image.shape) == 3:
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = test_image

        stat_features = fingerprint_core._extract_statistical_features(gray)

        assert isinstance(stat_features, np.ndarray)
        assert len(stat_features.shape) == 1
        assert np.all(np.isfinite(stat_features))

    def test_pattern_generation(self, fingerprint_core, test_image):
        """Test pattern generation."""
        features = fingerprint_core._extract_features_at_scale(test_image, 1.0)
        pattern = fingerprint_core._generate_pattern(features, test_image)

        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == fingerprint_core.fingerprint_size
        assert pattern.dtype == np.float32
        assert np.all((pattern >= 0) & (pattern <= 1))

    def test_error_correction(self, fingerprint_core):
        """Test error correction."""
        test_pattern = np.random.rand(*fingerprint_core.fingerprint_size)
        corrected = fingerprint_core._apply_error_correction(test_pattern)

        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == test_pattern.shape
        assert np.all(np.isfinite(corrected))

    def test_wavelet_comparison(self, fingerprint_core):
        """Test wavelet comparison."""
        fp1 = np.random.rand(*fingerprint_core.fingerprint_size)
        fp2 = fp1 + np.random.normal(0, 0.1, fp1.shape)  # Similar fingerprint

        similarity = fingerprint_core._compare_wavelets(fp1, fp2)

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

        # Compare with dissimilar fingerprint
        fp3 = np.random.rand(*fingerprint_core.fingerprint_size)
        diff_similarity = fingerprint_core._compare_wavelets(fp1, fp3)

        assert diff_similarity < similarity  # Should be less similar

    def test_fingerprint_comparison(self, fingerprint_core):
        """Test fingerprint comparison."""
        fp1 = np.random.rand(*fingerprint_core.fingerprint_size)
        fp2 = fp1 + np.random.normal(0, 0.1, fp1.shape)

        similarity, modifications = fingerprint_core.compare_fingerprints(fp1, fp2)

        assert isinstance(similarity, float)
        assert isinstance(modifications, list)
        assert 0 <= similarity <= 1

    @pytest.mark.parametrize("modification", [
        "compression", "geometric", "intensity"
    ])
    def test_modification_detection(self, fingerprint_core, test_image, modification):
        """Test specific modification detection."""
        original = fingerprint_core.generate_fingerprint(test_image)

        # Apply modifications
        modified = original.copy()
        if modification == "compression":
            modified = cv2.resize(cv2.resize(modified, (32, 32)),
                                  fingerprint_core.fingerprint_size)
        elif modification == "geometric":
            modified = cv2.resize(modified, (int(modified.shape[1] * 1.1),
                                             int(modified.shape[0] * 1.1)))[:-5, :-5]
        elif modification == "intensity":
            modified = modified * 1.2

        similarity, mods = fingerprint_core.compare_fingerprints(original, modified)

        assert modification in mods  # Should detect the applied modification
