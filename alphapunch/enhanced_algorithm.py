# enhanced_algorithm.py

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from alphapunch.amdf import AdaptiveMultiDomainFingerprinting
import os
import hashlib


class EnhancedAlphaPunch:
    def __init__(self, private_key, logger, fingerprint_size=(64, 64), embed_strength=0.15):
        self.private_key = private_key.encode()
        self.fingerprint_size = fingerprint_size
        self.logger = logger
        self.amdf = AdaptiveMultiDomainFingerprinting(fingerprint_size, embed_strength)
        self.similarity_threshold = 0.5  # Initial threshold
        self.min_threshold = 0.3
        self.max_threshold = 0.7
        self.recent_similarities = []

    def update_similarity_threshold(self, similarity):
        self.recent_similarities.append(similarity)
        if len(self.recent_similarities) > 20:
            self.recent_similarities.pop(0)

        mean_similarity = np.mean(self.recent_similarities)
        std_similarity = np.std(self.recent_similarities)

        # Adjust threshold based on recent statistics
        self.similarity_threshold = max(self.min_threshold,
                                        min(self.max_threshold, mean_similarity - 0.5 * std_similarity))
        self.logger.info(f"Updated similarity threshold: {self.similarity_threshold:.2f}")

    def train_verifier(self, authentic_pairs, fake_pairs):
        self.logger.info("Training verifier...")
        self.amdf.train_verifier(authentic_pairs, fake_pairs)
        self.logger.info("Verifier training complete.")

    def embed_fingerprint(self, image_input, output_path):
        self.logger.info("Starting enhanced fingerprint embedding process...")

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise ValueError("image_input must be either a file path or a numpy array")

        self.logger.info(f"Image loaded. Shape: {img.shape}")

        fingerprint = self.amdf.generate_fingerprint(img)
        embedded_img = self.amdf.embed_fingerprint(img, fingerprint)

        cv2.imwrite(output_path, embedded_img)
        self.logger.info(f"Fingerprinted image saved to {output_path}")

        psnr = peak_signal_noise_ratio(img, embedded_img)
        ssim = structural_similarity(img, embedded_img, channel_axis=2, data_range=255)
        self.logger.info(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

        return fingerprint, psnr, ssim

    def verify_fingerprint(self, image_input, original_fingerprint):
        self.logger.info("Starting fingerprint verification process...")

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError("image_input must be either a file path or a numpy array")

        similarity = self.amdf.verify_fingerprint(img, original_fingerprint)
        is_authentic = similarity > self.similarity_threshold

        self.logger.info(f"Fingerprint similarity: {similarity:.2%}")
        self.logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")

        # Adaptive thresholding
        self.update_similarity_threshold(similarity)

        return is_authentic, similarity, 1 - similarity  # normalized_hamming_distance

    def generate_key(self, seed):
        """Generate a unique key based on a seed and the private key."""
        combined = seed.encode() + self.private_key
        return hashlib.sha256(combined).hexdigest()

    def save_fingerprint(self, fingerprint, output_path):
        """Save the fingerprint to a file."""
        np.save(output_path, fingerprint)
        self.logger.info(f"Fingerprint saved to {output_path}")

    def load_fingerprint(self, input_path):
        """Load a fingerprint from a file."""
        fingerprint = np.load(input_path)
        self.logger.info(f"Fingerprint loaded from {input_path}")
        return fingerprint

    def batch_process(self, input_dir, output_dir):
        """Process all images in a directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"fp_{filename}")
                fingerprint_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_fingerprint.npy")

                fingerprint, psnr, ssim = self.embed_fingerprint(input_path, output_path)
                self.save_fingerprint(fingerprint, fingerprint_path)

                self.logger.info(f"Processed {filename}: PSNR = {psnr:.2f}, SSIM = {ssim:.4f}")

    def verify_batch(self, input_dir, fingerprint_dir):
        """Verify all images in a directory against their stored fingerprints."""
        results = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                fingerprint_path = os.path.join(fingerprint_dir, f"{os.path.splitext(filename)[0]}_fingerprint.npy")

                if os.path.exists(fingerprint_path):
                    original_fingerprint = self.load_fingerprint(fingerprint_path)
                    is_authentic, similarity, _ = self.verify_fingerprint(input_path, original_fingerprint)
                    results.append((filename, is_authentic, similarity))
                else:
                    self.logger.warning(f"No fingerprint found for {filename}")

        return results
