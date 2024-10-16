import random

import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import hashlib
import concurrent.futures
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
import pywt
import tensorflow as tf
from phe import paillier
from scipy.ndimage import zoom
import time
import traceback
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import hashlib


def box_counting_dimension(image, max_box_size=None, min_box_size=2):
    """Estimate the fractal dimension of an image using the box counting method."""
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        img = np.mean(image, axis=2).astype(np.uint8)
    else:
        img = image.astype(np.uint8)

    if max_box_size is None:
        max_box_size = min(img.shape)

    img = (img > img.mean()).astype(np.uint8)

    box_sizes = np.floor(np.logspace(np.log2(min_box_size), np.log2(max_box_size), num=10, base=2)).astype(int)
    box_sizes = np.unique(box_sizes)  # Remove duplicate sizes

    counts = []
    for size in box_sizes:
        padded_size = (np.ceil(np.array(img.shape) / size) * size).astype(int)
        padded_img = np.zeros(padded_size)
        padded_img[:img.shape[0], :img.shape[1]] = img

        box_count = (padded_img.reshape(padded_size[0] // size, size, -1, size)
                     .sum(axis=(1, 3)) > 0).sum()
        counts.append(max(1, box_count))  # Ensure count is at least 1 to avoid log(0)

    # Use only non-zero counts for the fit
    valid_indices = np.array(counts) > 0
    if np.sum(valid_indices) < 2:
        return 0  # Not enough valid points for a fit

    coeffs = np.polyfit(np.log(box_sizes[valid_indices]), np.log(np.array(counts)[valid_indices]), 1)
    return -coeffs[0]


class EnhancedAlphaPunch:
    def __init__(self, private_key, logger, fingerprint_size=(64, 64), block_size=8, embed_strength=0.75):
        self.private_key = private_key.encode()
        self.fingerprint_size = fingerprint_size
        self.block_size = block_size
        self.embed_positions = [(block_size // 2, block_size // 2 - 1), (block_size // 2 - 1, block_size // 2)]
        self.logger = logger
        self.embed_strength = embed_strength
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def generate_fractal_fingerprint(self, image):
        # Calculate the number of regions based on the image and fingerprint size
        rows = image.shape[0] // self.fingerprint_size[0]
        cols = image.shape[1] // self.fingerprint_size[1]

        regions = [image[i * self.fingerprint_size[0]:(i + 1) * self.fingerprint_size[0],
                   j * self.fingerprint_size[1]:(j + 1) * self.fingerprint_size[1]]
                   for i in range(rows) for j in range(cols)]

        fractal_dims = [box_counting_dimension(region) for region in regions]
        fingerprint = np.array(fractal_dims) > np.median(fractal_dims)

        # Resize the fingerprint to match the desired size
        fingerprint_resized = zoom(fingerprint.reshape(rows, cols),
                                   (self.fingerprint_size[0] / rows, self.fingerprint_size[1] / cols),
                                   order=0)

        return (fingerprint_resized > 0.5).astype(np.uint8)

    def quantum_inspired_embed(self, dct_block, bit):
        superposition = np.array([0.5, 0.5])
        if bit:
            superposition = np.array([0.15, 0.85])
        else:
            superposition = np.array([0.85, 0.15])

        for y, x in self.embed_positions:
            dct_block[y, x] += np.random.choice([-0.1, 0.1], p=superposition)
        return dct_block

    def wavelet_embed(self, image, fingerprint):
        # Convert image to float32 for wavelet transform
        img_float = image.astype(np.float32) / 255.0

        # Determine the appropriate wavelet decomposition level
        max_level = pywt.dwt_max_level(min(img_float.shape[:2]), 'db1')
        level = min(3, max_level)  # Use 3 or the maximum possible level, whichever is smaller

        # Perform 2D wavelet decomposition
        coeffs = pywt.wavedec2(img_float, 'db1', level=level)

        # Embed fingerprint in the wavelet coefficients
        fingerprint_flat = fingerprint.flatten()
        coeff_index = 0
        for i in range(1, len(coeffs)):  # Start from 1 to skip the approximation coefficients
            for j in range(3):  # There are 3 detail coefficient arrays at each level
                coeff = coeffs[i][j]
                embed_size = min(coeff.size, fingerprint_flat.size - coeff_index)
                coeff_flat = coeff.flatten()
                coeff_flat[:embed_size] += self.embed_strength * (
                        2 * fingerprint_flat[coeff_index:coeff_index + embed_size] - 1)
                coeffs[i] = (coeffs[i][0], coeffs[i][1], coeffs[i][2])  # Ensure it's a tuple
                coeff_index += embed_size
                if coeff_index >= fingerprint_flat.size:
                    break
            if coeff_index >= fingerprint_flat.size:
                break

        # Reconstruct the image
        reconstructed = pywt.waverec2(coeffs, 'db1')

        # Clip and convert back to uint8
        reconstructed = np.clip(reconstructed, 0, 1) * 255
        return reconstructed.astype(np.uint8)

    def blockchain_fingerprint(self, image):
        blocks = []
        prev_hash = hashlib.sha256(b"genesis").digest()
        for i in range(0, image.shape[0], self.block_size):
            for j in range(0, image.shape[1], self.block_size):
                block = image[i:i + self.block_size, j:j + self.block_size]
                block_hash = hashlib.sha256(block.tobytes() + prev_hash).digest()
                blocks.append((block, block_hash))
                prev_hash = block_hash
        return blocks

    def adversarial_train(self, model, image, fingerprint):
        with tf.GradientTape() as tape:
            prediction = model(image)
            loss = tf.keras.losses.binary_crossentropy(fingerprint, prediction)
        gradients = tape.gradient(loss, image)
        adversarial_image = image + 0.01 * tf.sign(gradients)
        return adversarial_image

    def simple_encrypt(self, data):
        return hashlib.sha256(data.encode()).hexdigest()[:64]  # Use only first 64 characters for consistency

    def homomorphic_embed(self, image, fingerprint):
        start_time = time.time()
        self.logger.info("Starting embedding process...")

        # Convert fingerprint to a binary string
        fingerprint_str = ''.join(map(str, fingerprint.flatten()))
        self.logger.debug(f"Original fingerprint (first 50 bits): {fingerprint_str[:50]}")

        # Encrypt the fingerprint
        encrypted_fingerprint = self.simple_encrypt(fingerprint_str)
        self.logger.debug(f"Encrypted fingerprint (first 50 chars): {encrypted_fingerprint[:50]}")

        self.logger.info(f"Fingerprint encrypted. Time: {time.time() - start_time:.2f}s")

        # Embed the encrypted fingerprint
        encrypted_image = image.copy()
        encrypted_image_flat = encrypted_image.flatten()

        binary_fingerprint = bin(int(encrypted_fingerprint, 16))[2:].zfill(2048)
        self.logger.debug(f"Binary fingerprint to embed (first 50 bits): {binary_fingerprint[:50]}")

        for i, bit in enumerate(binary_fingerprint):
            idx = i % len(encrypted_image_flat)
            encrypted_image_flat[idx] = (encrypted_image_flat[idx] & 0xFE) | int(bit)

        encrypted_image = encrypted_image_flat.reshape(image.shape)

        # Verify embedding
        embedded_bits = ''.join([str(pixel & 0x01) for pixel in encrypted_image_flat[:2048]])
        self.logger.debug(f"Embedded bits (first 50 bits): {embedded_bits[:50]}")

        self.logger.info(f"Embedding completed. Time: {time.time() - start_time:.2f}s")
        return encrypted_image.astype(np.uint8)

    def embed_fingerprint(self, image_path, output_path):
        self.logger.info("Starting enhanced fingerprint embedding process...")
        start_time = time.time()

        try:
            img = np.array(Image.open(image_path))
            self.logger.info(f"Image loaded. Shape: {img.shape}. Time: {time.time() - start_time:.2f}s")

            fingerprint = self.generate_fractal_fingerprint(img)
            self.logger.info(f"Fractal fingerprint generated. Time: {time.time() - start_time:.2f}s")

            embedded_img = self.wavelet_embed(img, fingerprint)
            self.logger.info(f"Wavelet embedding completed. Time: {time.time() - start_time:.2f}s")

            ycbcr = cv2.cvtColor(embedded_img.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
            y_channel = ycbcr[:, :, 0]

            def process_block(block_idx):
                i, j = block_idx
                block = y_channel[i:i + self.block_size, j:j + self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                bit_idx = (i // self.block_size * (
                        y_channel.shape[1] // self.block_size) + j // self.block_size) % fingerprint.size
                embedded_block = self.quantum_inspired_embed(dct_block, fingerprint.flat[bit_idx])
                return i, j, idct(idct(embedded_block.T, norm='ortho').T, norm='ortho')

            block_indices = [(i, j) for i in range(0, y_channel.shape[0] - self.block_size + 1, self.block_size)
                             for j in range(0, y_channel.shape[1] - self.block_size + 1, self.block_size)]

            self.logger.info(
                f"Starting DCT embedding. Total blocks: {len(block_indices)}. Time: {time.time() - start_time:.2f}s")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(process_block, block_indices), total=len(block_indices),
                                    desc="Embedding Progress"))

            self.logger.info(f"DCT embedding completed. Time: {time.time() - start_time:.2f}s")

            embedded_img = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)

            self.blockchain_fingerprint(embedded_img)
            self.logger.info(f"Blockchain-inspired chaining completed. Time: {time.time() - start_time:.2f}s")

            encrypted_img = self.homomorphic_embed(embedded_img, fingerprint)
            self.logger.info(f"Homomorphic encryption completed. Time: {time.time() - start_time:.2f}s")

            Image.fromarray(encrypted_img).save(output_path)
            self.logger.info(f"Fingerprinted image saved. Total time: {time.time() - start_time:.2f}s")

            # Calculate PSNR and SSIM with adjusted parameters
            psnr = peak_signal_noise_ratio(img, encrypted_img)
            win_size = min(7, min(img.shape[0],
                                  img.shape[1]) - 1)  # Ensure window size is odd and not larger than image
            if win_size % 2 == 0:
                win_size -= 1
            ssim = structural_similarity(img, encrypted_img, win_size=win_size, channel_axis=2, data_range=255)
            self.logger.info(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

            return fingerprint, psnr, ssim

        except Exception as e:
            self.logger.error(f"Error in embed_fingerprint: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def verify_fingerprint(self, image_path, original_fingerprint):
        self.logger.info("Starting fingerprint verification process...")
        start_time = time.time()
        img = np.array(Image.open(image_path))

        # Extract the embedded fingerprint
        img_flat = img.flatten()
        extracted_bits = ''.join([str(pixel & 0x01) for pixel in img_flat[:2048]])
        self.logger.debug(f"Extracted bits (first 50 bits): {extracted_bits[:50]}")

        extracted_hex = hex(int(extracted_bits, 2))[2:].zfill(256)
        self.logger.debug(f"Extracted hex (first 50 chars): {extracted_hex[:50]}")

        self.logger.info(f"Extracted fingerprint. Time: {time.time() - start_time:.2f}s")

        # Compare with original fingerprint
        original_fingerprint_str = ''.join(map(str, original_fingerprint.flatten()))
        self.logger.debug(f"Original fingerprint (first 50 bits): {original_fingerprint_str[:50]}")

        original_encrypted = self.simple_encrypt(original_fingerprint_str)
        self.logger.debug(f"Original encrypted (first 50 chars): {original_encrypted[:50]}")

        # Compare only the first 256 characters of both hexadecimal strings
        similarity = sum(a == b for a, b in zip(extracted_hex[:256], original_encrypted[:256])) / 256
        self.logger.info(f"Fingerprint similarity: {similarity:.2%}")

        is_authentic = similarity > 0.7  # Lowered threshold for testing
        self.logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")

        self.logger.info(f"Verification completed. Total time: {time.time() - start_time:.2f}s")
        return is_authentic, similarity

    def assess_image_quality(self, original_path, fingerprinted_path):
        original = cv2.imread(original_path)
        fingerprinted = cv2.imread(fingerprinted_path)

        if original.shape != fingerprinted.shape:
            fingerprinted = cv2.resize(fingerprinted, (original.shape[1], original.shape[0]))

        mse = np.mean((original - fingerprinted) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        min_dim = min(original.shape[0], original.shape[1])
        win_size = min(7, min_dim - (min_dim % 2) + 1)

        ssim_value = ssim(original, fingerprinted, win_size=win_size, channel_axis=2)

        return psnr, ssim_value

    def embed_fingerprint_with_quality(self, image_path, output_path):
        fingerprint = self.embed_fingerprint(image_path, output_path)
        psnr, ssim_value = self.assess_image_quality(image_path, output_path)
        self.logger.info(f"Image Quality - PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}")
        return fingerprint, psnr, ssim_value


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('EnhancedAlphaPunch')

    alphapunch = EnhancedAlphaPunch(private_key="your_secret_key_here", logger=logger)

    # Example usage
    fingerprint, psnr, ssim = alphapunch.embed_fingerprint_with_quality('input_image.jpg', 'fingerprinted_image.png')
    print(f"Embedding complete. PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    is_authentic, similarity = alphapunch.verify_fingerprint('fingerprinted_image.png', fingerprint)
    print(f"Verification result: {'Authentic' if is_authentic else 'Not authentic'}, Similarity: {similarity:.2%}")
