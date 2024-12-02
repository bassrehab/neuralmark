import hashlib
import logging
from typing import Tuple, List

import cv2
import numpy as np
import pywt
from scipy import stats
from scipy.fftpack import dct
from skimage.metrics import structural_similarity


class FingerprintCore:
    """Core fingerprinting functionality with enhanced security and robustness."""

    def __init__(self, private_key: str, config: dict, logger: logging.Logger):
        """Initialize fingerprint core with configuration."""
        # Initialize basic parameters
        self.logger = logger
        self.config = config

        # Generate secure key using provided private key
        self.private_key = self._generate_secure_key(private_key)

        # Initialize parameters from config
        self.fingerprint_size = tuple(config['algorithm']['fingerprint_size'])
        self.embed_strength = config['algorithm']['embed_strength']
        self.scales = config['algorithm'].get('scales', [1.0, 0.75, 0.5])

        # Initialize wavelet parameters
        self.wavelet = 'db1'  # Daubechies wavelets are good for image processing
        self.wavelet_level = 3

        # Initialize DCT weights based on private key
        self.dct_weights = self._generate_dct_weights()

        # Initialize feature weights
        self.feature_weights = config['algorithm']['feature_weights']

        # Initialize error correction parameters
        self.error_correction = config['algorithm']['error_correction']

        self.logger.debug("FingerprintCore initialized successfully")

    def _generate_secure_key(self, private_key: str) -> bytes:
        """Generate a secure key using PBKDF2."""
        salt = b'alphapunch_2024'  # You might want to make this configurable
        return hashlib.pbkdf2_hmac(
            'sha256',
            private_key.encode(),
            salt,
            100000  # Number of iterations
        )

    def _generate_dct_weights(self) -> np.ndarray:
        """Generate DCT weights based on private key with improved entropy."""
        # Use private key to seed numpy random
        np.random.seed(int.from_bytes(self.private_key[:4], byteorder='big'))

        # Generate random weights
        weights = np.random.rand(*self.fingerprint_size)

        # Apply DCT to add structure to the weights
        weights = dct(dct(weights.T, norm='ortho').T, norm='ortho')

        # Normalize weights
        weights = weights / np.max(np.abs(weights))

        return weights

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Generate robust fingerprint using multi-scale approach."""
        self.logger.debug("Generating fingerprint...")

        try:
            # Extract features at multiple scales
            features = []
            for scale in self.scales:
                scaled_features = self._extract_features_at_scale(image, scale)
                features.append(scaled_features)

            # Combine features
            combined_features = np.mean(features, axis=0)

            # Generate fingerprint pattern
            fingerprint = self._generate_pattern(combined_features, image)

            # Apply error correction
            if self.error_correction['enabled']:
                fingerprint = self._apply_error_correction(fingerprint)

            return fingerprint

        except Exception as e:
            self.logger.error(f"Error generating fingerprint: {str(e)}")
            raise

    def _extract_features_at_scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Extract features at a specific scale."""
        # Resize image
        height, width = image.shape[:2]
        scaled_size = (int(width * scale), int(height * scale))
        scaled_img = cv2.resize(image, scaled_size)

        # Convert to grayscale if needed
        if len(scaled_img.shape) == 3:
            gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = scaled_img

        # Extract DCT features
        dct_features = self._extract_dct_features(gray)

        # Extract wavelet features
        wavelet_features = self._extract_wavelet_features(gray)

        # Extract statistical features
        stat_features = self._extract_statistical_features(gray)

        # Combine features with weights
        combined = np.concatenate([
            self.feature_weights['dct'] * dct_features,
            self.feature_weights['wavelet'] * wavelet_features,
            self.feature_weights['vgg'] * stat_features
        ])

        return combined

    def _extract_dct_features(self, image: np.ndarray) -> np.ndarray:
        """Extract DCT features with improved robustness."""
        # Apply DCT
        dct_coeffs = dct(dct(image.T, norm='ortho').T, norm='ortho')

        # Select significant coefficients
        features = dct_coeffs[:8, :8].flatten()

        # Normalize features
        return features / (np.linalg.norm(features) + 1e-10)

    def _extract_wavelet_features(self, image: np.ndarray) -> np.ndarray:
        """Extract wavelet features with statistical properties."""
        # Wavelet decomposition
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.wavelet_level)

        features = []
        for level in coeffs[1:]:  # Skip approximation
            for detail in level:
                # Extract statistical features from each detail coefficient
                features.extend([
                    np.mean(np.abs(detail)),
                    np.std(detail),
                    stats.skew(detail.flatten()),
                    stats.kurtosis(detail.flatten())
                ])

        return np.array(features)

    def _extract_statistical_features(self, image: np.ndarray) -> np.ndarray:
        """Extract statistical features for improved robustness."""
        # Calculate local statistics
        features = []

        # Block processing
        block_size = 16
        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                block = image[i:i + block_size, j:j + block_size]
                if block.size > 0:
                    features.extend([
                        np.mean(block),
                        np.std(block),
                        stats.skew(block.flatten()),
                        stats.kurtosis(block.flatten())
                    ])

        return np.array(features)

    def _generate_pattern(self, features: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Generate fingerprint pattern with improved security."""
        # Create image hash
        img_hash = hashlib.sha256(image.tobytes()).digest()

        # Combine with private key
        combined_key = bytes([a ^ b for a, b in zip(self.private_key, img_hash)])

        # Generate pattern
        np.random.seed(int.from_bytes(combined_key[:4], byteorder='big'))
        pattern = np.random.rand(*self.fingerprint_size)

        # Apply feature modulation
        pattern = pattern * self.dct_weights

        # Normalize
        pattern = cv2.normalize(pattern, None, 0, 1, cv2.NORM_MINMAX)

        return pattern.astype(np.float32)

    def _apply_error_correction(self, fingerprint: np.ndarray) -> np.ndarray:
        """Apply error correction with adaptive parameters."""
        # Get parameters from config
        kernel_size = self.error_correction['gaussian_kernel_size']
        sigma = self.error_correction['gaussian_sigma']

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply Gaussian smoothing
        smoothed = cv2.GaussianBlur(
            fingerprint,
            (kernel_size, kernel_size),
            sigma
        )

        # Enhance contrast
        enhanced = cv2.normalize(smoothed, None, 0, 1, cv2.NORM_MINMAX)

        return enhanced

    def compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> Tuple[float, List[str]]:
        """
        Compare fingerprints with comprehensive similarity metrics.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Tuple[float, List[str]]: Similarity score and list of detected modifications
        """
        modifications = []

        try:
            # Normalize fingerprints
            fp1_norm = cv2.normalize(fp1, None, 0, 1, cv2.NORM_MINMAX)
            fp2_norm = cv2.normalize(fp2, None, 0, 1, cv2.NORM_MINMAX)

            # Calculate NCC
            ncc = float(cv2.matchTemplate(
                fp1_norm.astype(np.float32),
                fp2_norm.astype(np.float32),
                cv2.TM_CCORR_NORMED
            )[0][0])  # Convert numpy.float32 to float

            # Calculate SSIM
            ssim = float(structural_similarity(fp1_norm, fp2_norm, data_range=1.0))

            # Frequency domain comparison
            f1 = np.fft.fft2(fp1_norm)
            f2 = np.fft.fft2(fp2_norm)
            freq_diff = np.abs(f1 - f2)

            # Detect modifications
            if np.mean(freq_diff) > 0.5:
                modifications.append("JPEG compression")
            if np.std(freq_diff) > 0.3:
                modifications.append("Geometric transformation")
            if np.abs(np.mean(fp1) - np.mean(fp2)) > 0.2:
                modifications.append("Intensity modification")

            # Calculate wavelet similarity
            wavelet_sim = float(self._compare_wavelets(fp1_norm, fp2_norm))  # Convert to float

            # Combine similarities with weights
            similarity = float(0.4 * ncc + 0.3 * ssim + 0.3 * wavelet_sim)  # Ensure float output

            return similarity, modifications

        except Exception as e:
            self.logger.error(f"Error comparing fingerprints: {str(e)}")
            raise

    def _compare_wavelets(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compare fingerprints in wavelet domain."""
        # Decompose both fingerprints
        coeffs1 = pywt.wavedec2(fp1, self.wavelet, level=self.wavelet_level)
        coeffs2 = pywt.wavedec2(fp2, self.wavelet, level=self.wavelet_level)

        # Compare coefficients
        similarities = []
        for c1, c2 in zip(coeffs1, coeffs2):
            if isinstance(c1, tuple):
                # Detail coefficients
                for d1, d2 in zip(c1, c2):
                    sim = np.corrcoef(d1.flatten(), d2.flatten())[0, 1]
                    similarities.append(sim)
            else:
                # Approximation coefficients
                sim = np.corrcoef(c1.flatten(), c2.flatten())[0, 1]
                similarities.append(sim)

        return np.mean(similarities)
