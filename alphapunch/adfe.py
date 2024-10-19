# adfe.py

import numpy as np
import cv2
from scipy.stats import skew
from skimage.measure import block_reduce
from pywt import wavedec2, waverec2


class AdaptiveDimensionalFingerprintEncoding:
    def __init__(self, base_size=(64, 64)):
        self.base_size = base_size

    def calculate_fractal_dimension(self, image):
        # Simplified box-counting method
        sizes = np.array([2, 4, 8, 16, 32, 64])
        counts = []
        for size in sizes:
            count = np.sum(block_reduce(image, (size, size), np.max) > 0)
            counts.append(count)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def generate_adaptive_fingerprint(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fd = self.calculate_fractal_dimension(gray_image)
        h, w = image.shape[:2]
        aspect_ratio = w / h

        # Adjust fingerprint size based on fractal dimension and aspect ratio
        fp_h = int(self.base_size[0] * (1 + fd * 0.1))
        fp_w = int(fp_h * aspect_ratio)

        # Generate fingerprint using image statistics
        fingerprint = np.random.rand(fp_h, fp_w) < (np.mean(gray_image) / 255)
        return fingerprint.astype(np.float32)

    def embed_fingerprint(self, image, fingerprint):
        # Perform wavelet decomposition
        coeffs = wavedec2(image, 'db1', level=3)

        # Embed in multiple subbands
        for i in range(1, len(coeffs)):
            subband = coeffs[i]
            for j in range(3):
                subband_shape = subband[j].shape
                resized_fingerprint = cv2.resize(fingerprint, (subband_shape[1], subband_shape[0]))
                subband[j] += 0.1 * resized_fingerprint

        # Reconstruct image
        embedded_image = waverec2(coeffs, 'db1')
        return np.clip(embedded_image, 0, 255).astype(np.uint8)

    def extract_fingerprint(self, image):
        # Convert image to grayscale if it's color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform wavelet decomposition
        coeffs = wavedec2(image, 'db1', level=3)

        # Extract from multiple subbands
        extracted = np.zeros_like(coeffs[1][0], dtype=float)
        for i in range(1, len(coeffs)):
            subband = coeffs[i]
            for j in range(3):
                resized_subband = cv2.resize(subband[j], (extracted.shape[1], extracted.shape[0]))
                extracted += resized_subband

        # Threshold to get binary fingerprint
        return (extracted > np.median(extracted)).astype(np.float32)

    def verify_fingerprint(self, image, original_fingerprint):
        extracted = self.extract_fingerprint(image)

        # Ensure original_fingerprint is 2D
        if len(original_fingerprint.shape) == 3:
            original_fingerprint = original_fingerprint[:, :, 0]

        resized_original = cv2.resize(original_fingerprint, (extracted.shape[1], extracted.shape[0]))

        similarity = np.mean(extracted == resized_original)
        return similarity > 0.7, similarity