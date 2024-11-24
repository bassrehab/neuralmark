import logging
from typing import Tuple, List

import cv2
import numpy as np
import pywt
from .core.amdf import AdaptiveMultiDomainFingerprinting


class AlphaPunch:
    def __init__(self, private_key: str, logger: logging.Logger, config: dict):
        """Initialize EnhancedAlphaPunch with configuration."""
        self.private_key = private_key
        self.logger = logger
        self.config = config

        # Initialize AMDF
        self.amdf = AdaptiveMultiDomainFingerprinting(
            config=config,
            logger=logger
        )

        # Initialize parameters
        self.fingerprint_size = tuple(config['algorithm']['fingerprint_size'])
        self.embed_strength = config['algorithm']['embed_strength']

        if self.logger:
            self.logger.info("EnhancedAlphaPunch initialized")

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Generate fingerprint for an image."""
        self.logger.debug("Generating fingerprint...")

        try:
            fingerprint = self.amdf.generate_fingerprint(image)
            self.logger.debug("Fingerprint generated successfully")
            return fingerprint

        except Exception as e:
            self.logger.error(f"Error generating fingerprint: {str(e)}")
            raise

    def _calculate_embedding_mask(self, image: np.ndarray) -> np.ndarray:
        """Calculate embedding mask with proper type handling."""
        try:
            # Ensure image is uint8
            image = image.astype(np.uint8)

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Calculate edge map
            edges = cv2.Canny(gray, 100, 200)

            # Calculate texture map using Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture = np.abs(laplacian)
            texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Combine masks
            combined = cv2.addWeighted(edges, 0.5, texture, 0.5, 0)

            # Apply Gaussian blur
            mask = cv2.GaussianBlur(combined, (5, 5), 0)

            return mask

        except Exception as e:
            self.logger.error(f"Error in _calculate_embedding_mask: {str(e)}")
            raise

    def embed_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
        """Embed fingerprint into image with proper type handling."""
        try:
            # Ensure image is uint8
            image = image.astype(np.uint8)

            # Convert fingerprint to proper range and type
            fingerprint = (fingerprint * 255).astype(np.uint8)

            # Calculate embedding mask with proper type handling
            mask = self._calculate_embedding_mask(image)

            # Prepare fingerprint
            fingerprint_resized = cv2.resize(fingerprint, (image.shape[1], image.shape[0]))
            fp_prepared = fingerprint_resized * (mask / 255.0) * self.embed_strength

            # Embed in wavelet domain
            embedded = np.zeros_like(image, dtype=np.float32)

            for i in range(3):
                # Convert to float32 for wavelet transform
                channel = image[:, :, i].astype(np.float32)
                coeffs = pywt.wavedec2(channel, 'db1', level=3)
                fp_coeffs = pywt.wavedec2(fp_prepared, 'db1', level=3)

                # Modify coefficients
                modified_coeffs = list(coeffs)
                for j in range(len(coeffs)):
                    if j == 0:  # Approximation coefficients
                        c_shape = modified_coeffs[j].shape
                        fp_shape = fp_coeffs[j].shape
                        if c_shape != fp_shape:
                            fp_coeffs[j] = cv2.resize(fp_coeffs[j],
                                                      (c_shape[1], c_shape[0]))
                        modified_coeffs[j] = coeffs[j] + fp_coeffs[j]
                    else:  # Detail coefficients
                        modified_coeffs[j] = tuple(
                            c + cv2.resize(f, (c.shape[1], c.shape[0])) * 0.5
                            for c, f in zip(coeffs[j], fp_coeffs[j])
                        )

                # Reconstruct
                embedded[:, :, i] = pywt.waverec2(modified_coeffs, 'db1')

            # Normalize and convert back to uint8
            embedded = np.clip(embedded, 0, 255).astype(np.uint8)
            return embedded

        except Exception as e:
            self.logger.error(f"Error in embed_fingerprint: {str(e)}")
            raise

    def extract_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Extract fingerprint from image."""
        self.logger.debug("Extracting fingerprint...")

        try:
            fingerprint = self.amdf.extract_fingerprint(image)
            self.logger.debug("Fingerprint extracted successfully")
            return fingerprint

        except Exception as e:
            self.logger.error(f"Error extracting fingerprint: {str(e)}")
            raise

    def compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> Tuple[float, List[str]]:
        """Compare two fingerprints."""
        self.logger.debug("Comparing fingerprints...")

        try:
            similarity, modifications = self.amdf.compare_fingerprints(fp1, fp2)
            self.logger.debug(f"Fingerprint comparison complete. Similarity: {similarity:.4f}")
            return similarity, modifications

        except Exception as e:
            self.logger.error(f"Error comparing fingerprints: {str(e)}")
            raise

    def verify_fingerprint(self, image: np.ndarray, original_fingerprint: np.ndarray) -> Tuple[bool, float, List[str]]:
        """Verify fingerprint with proper error handling."""
        try:
            # Extract fingerprint
            extracted_fp = self.extract_fingerprint(image)

            # Ensure fingerprints have same dimensions
            if extracted_fp.shape != original_fingerprint.shape:
                h, w = original_fingerprint.shape[:2]
                extracted_fp = cv2.resize(extracted_fp, (w, h))

            # Compare fingerprints
            similarity_result = self.amdf.compare_fingerprints(extracted_fp, original_fingerprint)

            if isinstance(similarity_result, tuple):
                similarity, modifications = similarity_result
            else:
                similarity = float(similarity_result)
                modifications = []

            # Get threshold
            threshold = self.config['algorithm']['similarity_threshold']

            # Determine authenticity
            is_authentic = similarity > threshold

            return is_authentic, similarity, modifications

        except Exception as e:
            self.logger.error(f"Error in verify_fingerprint: {str(e)}")
            raise

    def train_verifier(self, authentic_pairs, fake_pairs):
        """Train the verifier with authentic and fake pairs."""
        self.logger.info("Training verifier...")

        try:
            self.amdf.train_verifier(authentic_pairs, fake_pairs)
            self.logger.info("Verifier training complete")

        except Exception as e:
            self.logger.error(f"Error training verifier: {str(e)}")
            raise
