# enhanced_algorithm.py

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from alphapunch.amdf import AdaptiveMultiDomainFingerprinting
import os
import hashlib


class EnhancedAlphaPunch:
    def __init__(self, private_key, logger, config):
        self.private_key = private_key.encode()
        self.fingerprint_size = tuple(config['algorithm']['fingerprint_size'])
        self.logger = logger
        self.config = config  # Store the config
        self.amdf = AdaptiveMultiDomainFingerprinting(
            fingerprint_size=self.fingerprint_size,
            embed_strength=config['algorithm']['embed_strength'],
            config=config
        )
        self.similarity_threshold = config['algorithm']['similarity_threshold']

        # Set random seed for reproducibility
        np.random.seed(hash(private_key) % 2 ** 32)

    def update_similarity_threshold(self, similarity):
        self.recent_similarities.append(similarity)
        if len(self.recent_similarities) > self.adaptive_window:
            self.recent_similarities.pop(0)

        if len(self.recent_similarities) >= 3:
            # Sort similarities to find median
            sorted_sims = sorted(self.recent_similarities)
            median = sorted_sims[len(sorted_sims) // 2]

            # Adjust threshold to be slightly below median for better acceptance
            self.similarity_threshold = max(
                self.min_threshold,
                min(self.max_threshold, median * 0.85)
            )

        self.logger.info(f"Updated similarity threshold: {self.similarity_threshold:.2f}")

    def train_verifier(self, authentic_pairs, fake_pairs):
        self.logger.info("Training verifier...")
        self.amdf.train_verifier(authentic_pairs, fake_pairs)
        self.logger.info("Verifier training complete.")

    def embed_fingerprint(self, image_input, output_path):
        """Embed fingerprint in image."""
        self.logger.info("Starting fingerprint embedding process...")

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

    # enhanced_algorithm.py

    def verify_fingerprint(self, image_input, original_fingerprint):
        """Enhanced verification with stricter checks."""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError("image_input must be either a file path or a numpy array")

        ver_config = self.config['algorithm']['verification']

        # Extract and normalize fingerprints
        extracted = self.amdf.extract_fingerprint(img)
        extracted_norm = cv2.normalize(extracted, None, 0, 1, cv2.NORM_MINMAX)
        original_norm = cv2.normalize(original_fingerprint, None, 0, 1, cv2.NORM_MINMAX)

        # 1. Normalized Cross-Correlation (NCC)
        ncc = cv2.matchTemplate(
            extracted_norm.astype(np.float32),
            original_norm.astype(np.float32),
            cv2.TM_CCORR_NORMED
        )[0][0]

        # 2. Structural Similarity (SSIM)
        ssim = structural_similarity(extracted_norm, original_norm, data_range=1.0)

        # 3. Frequency Domain Similarity
        f1 = np.fft.fft2(extracted_norm)
        f2 = np.fft.fft2(original_norm)
        freq_sim = np.abs(np.corrcoef(np.abs(f1).flatten(), np.abs(f2).flatten())[0, 1])

        # 4. Edge Similarity
        edges_extracted = cv2.Sobel(extracted_norm, cv2.CV_64F, 1, 1)
        edges_original = cv2.Sobel(original_norm, cv2.CV_64F, 1, 1)
        edge_sim = structural_similarity(
            edges_extracted,
            edges_original,
            data_range=np.max(edges_extracted) - np.min(edges_extracted)
        )

        # Calculate weighted similarity
        similarity = (
                ncc * ver_config['ncc_weight'] +
                ssim * ver_config['ssim_weight'] +
                freq_sim * ver_config['freq_weight'] +
                edge_sim * ver_config['edge_weight']
        )

        # Apply penalties for low individual scores
        if ncc < ver_config['min_ncc_score'] or ssim < ver_config['min_ssim_score']:
            similarity *= ver_config['penalty_factor']
            self.logger.debug(f"Applied penalty. Original similarity: {similarity / ver_config['penalty_factor']:.2%}, "
                              f"After penalty: {similarity:.2%}")

        # Final authentication decision
        is_authentic = similarity > self.similarity_threshold

        # Detailed logging
        self.logger.info(f"NCC: {ncc:.2%}")
        self.logger.info(f"SSIM: {ssim:.2%}")
        self.logger.info(f"Frequency similarity: {freq_sim:.2%}")
        self.logger.info(f"Edge similarity: {edge_sim:.2%}")
        self.logger.info(f"Final similarity: {similarity:.2%}")
        self.logger.info(f"Threshold: {self.similarity_threshold:.2%}")
        self.logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")

        return is_authentic, similarity, 1 - similarity

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
