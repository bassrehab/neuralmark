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

    def _verify_dimensions(self, array: np.ndarray, target_shape: tuple, name: str = "array") -> np.ndarray:
        """Verify and correct array dimensions if needed."""
        if array.shape[:2] != target_shape[:2]:
            self.logger.debug(f"Resizing {name} from {array.shape} to match {target_shape}")
            return cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        return array

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Generate fingerprint for an image."""
        self.logger.debug("Generating fingerprint...")

        try:
            if self.use_attention:
                # Get base fingerprint
                fingerprint = self.amdf.generate_fingerprint(image)

                # Enhance with attention
                enhanced_fp, _ = self.attention_enhancer.enhance_fingerprint(image, fingerprint)

                # Ensure correct size
                enhanced_fp = cv2.resize(enhanced_fp, self.fingerprint_size)
                return enhanced_fp
            else:
                fingerprint = self.amdf.generate_fingerprint(image)
                # Ensure correct size
                fingerprint = cv2.resize(fingerprint, self.fingerprint_size)
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
        """Embed fingerprint into image with proper size handling."""
        try:
            # Ensure image is uint8
            image = image.astype(np.uint8)

            # Convert fingerprint to proper range and type
            fingerprint = cv2.normalize(fingerprint, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Calculate embedding mask
            mask = self._calculate_embedding_mask(image)

            # Get exact dimensions
            h, w = image.shape[:2]

            # Ensure fingerprint and mask match image dimensions exactly
            fingerprint_resized = cv2.resize(fingerprint, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Prepare fingerprint for embedding
            fp_prepared = fingerprint_resized * mask * self.embed_strength

            # Ensure fp_prepared matches image channels
            if len(image.shape) == 3 and len(fp_prepared.shape) == 2:
                fp_prepared = np.stack([fp_prepared] * 3, axis=-1)

            # Initialize output array with exact dimensions
            embedded = np.zeros_like(image, dtype=np.float32)

            # Process each channel
            channels = 3 if len(image.shape) == 3 else 1
            for i in range(channels):
                # Extract channel
                channel = image[..., i] if channels == 3 else image
                channel = channel.astype(np.float32)

                # Apply wavelet transform
                coeffs = pywt.wavedec2(channel, 'db1', level=3)

                # Get fingerprint coefficients
                fp_channel = fp_prepared[..., i] if channels == 3 else fp_prepared
                fp_coeffs = pywt.wavedec2(fp_channel, 'db1', level=3)

                # Modify coefficients
                modified_coeffs = list(coeffs)
                for j in range(len(coeffs)):
                    if j == 0:  # Approximation coefficients
                        c_shape = modified_coeffs[j].shape
                        fp_shape = fp_coeffs[j].shape
                        if c_shape != fp_shape:
                            # Use exact resize to match shapes
                            fp_coeffs[j] = cv2.resize(fp_coeffs[j],
                                                      (c_shape[1], c_shape[0]),
                                                      interpolation=cv2.INTER_LINEAR)
                        modified_coeffs[j] = coeffs[j] + fp_coeffs[j]
                    else:  # Detail coefficients
                        modified_coeffs[j] = tuple(
                            c + cv2.resize(f, (c.shape[1], c.shape[0]),
                                           interpolation=cv2.INTER_LINEAR) * 0.5
                            for c, f in zip(coeffs[j], fp_coeffs[j])
                        )

                # Reconstruct with exact dimensions
                reconstructed = pywt.waverec2(modified_coeffs, 'db1')

                # Ensure reconstructed image matches original dimensions
                if reconstructed.shape != (h, w):
                    reconstructed = cv2.resize(reconstructed, (w, h), interpolation=cv2.INTER_LINEAR)

                # Assign to output
                if channels == 3:
                    embedded[..., i] = reconstructed
                else:
                    embedded = reconstructed

            # Normalize and convert back to uint8
            embedded = np.clip(embedded, 0, 255).astype(np.uint8)

            # Final size verification
            if embedded.shape[:2] != image.shape[:2]:
                embedded = cv2.resize(embedded, (w, h), interpolation=cv2.INTER_LINEAR)

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
