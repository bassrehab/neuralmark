import logging
from typing import Tuple, List

import cv2
import numpy as np
import pywt
import tensorflow as tf
from skimage.metrics import structural_similarity

from .core.amdf import AdaptiveMultiDomainFingerprinting
from .core.neural_attention import NeuralAttentionEnhancer


class AlphaPunch:
    def __init__(self, private_key: str, logger: logging.Logger, config: dict):
        """Initialize EnhancedAlphaPunch with neural attention."""
        self.private_key = private_key
        self.logger = logger
        self.config = config

        # Initialize AMDF
        self.amdf = AdaptiveMultiDomainFingerprinting(
            config=config,
            logger=logger
        )

        # Initialize neural attention enhancer if enabled
        self.use_attention = config['algorithm'].get('neural_attention', {}).get('enabled', False)
        if self.use_attention:
            self.attention_enhancer = NeuralAttentionEnhancer(config, logger)
            self.logger.info("Neural attention enhancement initialized")

        # Initialize parameters
        self.fingerprint_size = tuple(config['algorithm']['fingerprint_size'])
        self.embed_strength = config['algorithm']['embed_strength']

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Generate fingerprint for an image with optional neural attention enhancement."""
        self.logger.debug("Generating fingerprint...")

        try:
            # Generate base fingerprint using AMDF
            fingerprint = self.amdf.generate_fingerprint(image)

            if self.use_attention:
                # Enhance fingerprint using neural attention
                self.logger.debug("Applying neural attention enhancement...")
                enhanced_fingerprint, _ = self.attention_enhancer.enhance_fingerprint(
                    image, fingerprint
                )
                self.logger.debug("Neural attention enhancement applied successfully")
                return enhanced_fingerprint

            return fingerprint

        except Exception as e:
            self.logger.error(f"Error generating fingerprint: {str(e)}")
            raise

    def _calculate_embedding_mask(self, image: np.ndarray) -> np.ndarray:
        """Calculate embedding mask with proper size handling."""
        try:
            # Ensure image is uint8
            image = image.astype(np.uint8)

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            height, width = gray.shape[:2]

            if self.use_attention:
                try:
                    # Get attention-based mask
                    dummy_fingerprint = np.zeros(self.fingerprint_size)
                    _, attention_mask = self.attention_enhancer.enhance_fingerprint(
                        image, dummy_fingerprint
                    )

                    # Resize attention mask to match image size
                    attention_mask = cv2.resize(
                        attention_mask.astype(np.float32),
                        (width, height)
                    )
                    attention_mask = cv2.normalize(
                        attention_mask, None, 0, 1, cv2.NORM_MINMAX
                    )
                except Exception as e:
                    self.logger.error(f"Error in attention mask: {str(e)}")
                    attention_mask = np.ones((height, width), dtype=np.float32)

                # Calculate traditional mask
                edges = cv2.Canny(gray, 100, 200)
                texture = cv2.Laplacian(gray, cv2.CV_64F)
                texture = np.abs(texture)
                texture = cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Ensure edges are float32 and normalized
                edges = edges.astype(np.float32) / 255.0

                # Combine traditional components
                traditional_mask = cv2.addWeighted(edges, 0.5, texture, 0.5, 0)
                traditional_mask = cv2.GaussianBlur(traditional_mask, (5, 5), 0)

                # Combine with attention mask
                combined_mask = cv2.addWeighted(
                    attention_mask, 0.6,
                    traditional_mask, 0.4,
                    0
                )

                return combined_mask
            else:
                # Traditional mask only
                edges = cv2.Canny(gray, 100, 200)
                texture = cv2.Laplacian(gray, cv2.CV_64F)
                texture = np.abs(texture)
                texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                combined = cv2.addWeighted(edges, 0.5, texture, 0.5, 0)
                return cv2.GaussianBlur(combined, (5, 5), 0) / 255.0

        except Exception as e:
            self.logger.error(f"Error in _calculate_embedding_mask: {str(e)}")
            # Return uniform mask as fallback
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)


    def embed_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
        """Embed fingerprint into image with neural attention enhancement."""
        try:
            # Ensure image is uint8
            image = image.astype(np.uint8)

            # Convert fingerprint to proper range and type
            fingerprint = (fingerprint * 255).astype(np.uint8)

            # Calculate embedding mask
            mask = self._calculate_embedding_mask(image)

            # Prepare fingerprint
            fingerprint_resized = cv2.resize(fingerprint, (image.shape[1], image.shape[0]))
            fp_prepared = fingerprint_resized * mask * self.embed_strength

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
                            c + cv2.resize(f, (c.shape[1], c.shape[0])) * mask * 0.5
                            for c, f in zip(coeffs[j], fp_coeffs[j])
                        )

                # Reconstruct
                embedded[:, :, i] = pywt.waverec2(modified_coeffs, 'db1')

            # Normalize and convert back to uint8
            embedded = np.clip(embedded, 0, 255).astype(np.uint8)

            # Apply post-processing if neural attention is enabled
            if self.use_attention:
                embedded = self._post_process_embedded(embedded, image)

            return embedded

        except Exception as e:
            self.logger.error(f"Error in embed_fingerprint: {str(e)}")
            raise

    def _post_process_embedded(self, embedded: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply post-processing to maintain image quality."""
        try:
            # Convert to LAB color space
            embedded_lab = cv2.cvtColor(embedded, cv2.COLOR_BGR2LAB)
            original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

            # Preserve original luminance in areas of high detail
            l_embedded, a, b = cv2.split(embedded_lab)
            l_original = cv2.split(original_lab)[0]

            # Calculate detail mask
            detail_mask = cv2.Laplacian(l_original, cv2.CV_64F)
            detail_mask = np.abs(detail_mask)
            detail_mask = cv2.normalize(detail_mask, None, 0, 1, cv2.NORM_MINMAX)

            # Blend luminance channels
            l_final = cv2.addWeighted(
                l_embedded.astype(float), 1 - detail_mask,
                l_original.astype(float), detail_mask, 0
            )

            # Merge channels and convert back
            result = cv2.merge([l_final.astype(np.uint8), a, b])
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        except Exception as e:
            self.logger.error(f"Error in post-processing: {str(e)}")
            return embedded

    def extract_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Extract fingerprint with neural attention guidance."""
        self.logger.debug("Extracting fingerprint...")

        try:
            if self.use_attention:
                # Get attention mask for extraction
                attention_mask = self._calculate_embedding_mask(image)
                # Extract with attention guidance
                fingerprint = self.amdf.extract_fingerprint(image * attention_mask[..., np.newaxis])
            else:
                fingerprint = self.amdf.extract_fingerprint(image)

            self.logger.debug("Fingerprint extracted successfully")
            return fingerprint

        except Exception as e:
            self.logger.error(f"Error extracting fingerprint: {str(e)}")
            raise

    def compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> Tuple[float, List[str]]:
        """Compare fingerprints with attention-weighted comparison."""
        self.logger.debug("Comparing fingerprints...")

        try:
            if self.use_attention:
                # Get attention weights for comparison
                attention_weights = self.attention_enhancer.get_comparison_weights(fp1, fp2)

                # Apply attention weights to comparison
                similarity, modifications = self.amdf.compare_fingerprints(
                    fp1 * attention_weights,
                    fp2 * attention_weights
                )
            else:
                similarity, modifications = self.amdf.compare_fingerprints(fp1, fp2)

            self.logger.debug(f"Fingerprint comparison complete. Similarity: {similarity:.4f}")
            return similarity, modifications

        except Exception as e:
            self.logger.error(f"Error comparing fingerprints: {str(e)}")
            raise

    def verify_fingerprint(self, image: np.ndarray, original_fingerprint: np.ndarray) -> Tuple[bool, float, List[str]]:
        """Verify fingerprint with attention-enhanced verification."""
        try:
            # Extract fingerprint
            extracted_fp = self.extract_fingerprint(image)

            # Ensure fingerprints have same dimensions
            if extracted_fp.shape != original_fingerprint.shape:
                h, w = original_fingerprint.shape[:2]
                extracted_fp = cv2.resize(extracted_fp, (w, h))

            # Compare fingerprints
            similarity, modifications = self.compare_fingerprints(extracted_fp, original_fingerprint)

            # Calculate adaptive threshold if using attention
            if self.use_attention:
                base_threshold = self.config['algorithm']['similarity_threshold']
                # Adjust threshold based on image complexity
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                complexity = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
                threshold = base_threshold * (1 - complexity * 0.2)
            else:
                threshold = self.config['algorithm']['similarity_threshold']

            # Determine authenticity
            is_authentic = similarity > threshold

            return is_authentic, similarity, modifications

        except Exception as e:
            self.logger.error(f"Error in verify_fingerprint: {str(e)}")
            raise

    def train_verifier(self, authentic_pairs, fake_pairs):
        """Train the verifier with neural attention if enabled."""
        self.logger.info("Training verifier...")

        try:
            if self.use_attention:
                # Train attention model first
                self.attention_enhancer.train(authentic_pairs, fake_pairs)

            # Train AMDF verifier
            self.amdf.train_verifier(authentic_pairs, fake_pairs)
            self.logger.info("Verifier training complete")

        except Exception as e:
            self.logger.error(f"Error training verifier: {str(e)}")
            raise
