import os
import time

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
from numba import jit


class EnhancedAlphaPunch:
    def __init__(self, private_key, logger, fingerprint_size=(64, 64), block_size=8, embed_strength=0.75,
                 use_gpu=False):
        self.private_key = private_key.encode()
        self.fingerprint_size = fingerprint_size
        self.block_size = block_size
        self.embed_positions = [(block_size // 2, block_size // 2 - 1), (block_size // 2 - 1, block_size // 2)]
        self.logger = logger
        self.embed_strength = embed_strength
        self.use_gpu = use_gpu

    def generate_fingerprint(self):
        np.random.seed(int.from_bytes(hashlib.sha256(self.private_key).digest(), byteorder='big') % 2 ** 32)
        fingerprint = np.random.randint(0, 2, self.fingerprint_size).astype(np.uint8)
        # Apply repetition code (3x repetition)
        fingerprint = np.repeat(fingerprint, 3)
        self.logger.debug(f"Generated fingerprint shape: {fingerprint.shape}")
        self.logger.debug(f"Fingerprint sample: {fingerprint[:15]}")
        return fingerprint

    def encrypt_fingerprint(self, fingerprint, salt):
        key = hashlib.sha256(self.private_key + salt).digest()
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted = cipher.encrypt(pad(fingerprint.tobytes(), AES.block_size))
        return iv + encrypted

    def decrypt_fingerprint(self, encrypted_fingerprint, salt):
        key = hashlib.sha256(self.private_key + salt).digest()
        iv = encrypted_fingerprint[:AES.block_size]
        encrypted = encrypted_fingerprint[AES.block_size:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        try:
            decrypted = cipher.decrypt(encrypted)
            unpadded = unpad(decrypted, AES.block_size)
        except ValueError:
            self.logger.warning("Padding error during decryption. Using raw decrypted data.")
            unpadded = decrypted
        decrypted_fingerprint = np.frombuffer(unpadded, dtype=np.uint8)
        self.logger.debug(f"Decrypted fingerprint shape: {decrypted_fingerprint.shape}")
        self.logger.debug(f"Decrypted fingerprint sample: {decrypted_fingerprint[:15]}")
        return decrypted_fingerprint

    @staticmethod
    @jit(nopython=True)
    def rgb_to_ycbcr(rgb):
        rgb = rgb.astype(np.float32) / 255.0
        transform = np.array([[0.299, 0.587, 0.114],
                              [-0.168736, -0.331264, 0.5],
                              [0.5, -0.418688, -0.081312]])
        ycbcr = rgb.dot(transform.T)
        ycbcr[:, :, 1:] += 0.5
        return np.clip(ycbcr, 0, 1) * 255

    @staticmethod
    @jit(nopython=True)
    def ycbcr_to_rgb(ycbcr):
        ycbcr = ycbcr.astype(np.float32) / 255.0
        ycbcr[:, :, 1:] -= 0.5
        transform = np.array([[1.0, 0.0, 1.402],
                              [1.0, -0.344136, -0.714136],
                              [1.0, 1.772, 0.0]])
        rgb = ycbcr.dot(transform.T)
        return np.clip(rgb, 0, 1) * 255

    def embed_in_dct_block(self, dct_block, bit):
        for y, x in self.embed_positions:
            original_value = dct_block[y, x]
            dct_block[y, x] = dct_block[y, x] + self.embed_strength if bit else dct_block[y, x] - self.embed_strength
            self.logger.debug(
                f"Embedding bit {bit}. Position: ({y},{x}), Original value: {original_value:.4f}, New value: {dct_block[y, x]:.4f}")
        return dct_block

    def extract_from_dct_block(self, dct_block):
        bits = []
        for y, x in self.embed_positions:
            extracted_bit = dct_block[y, x] > 0
            bits.append(extracted_bit)
            self.logger.debug(f"Extracted bit: {extracted_bit}, Position: ({y},{x}), DCT value: {dct_block[y, x]:.4f}")
        return bits

    def embed_fingerprint(self, image_path, output_path):
        self.logger.info("Starting fingerprint embedding process...")
        img = np.array(Image.open(image_path))
        ycbcr = self.rgb_to_ycbcr(img)

        self.logger.info("Generating and encrypting fingerprint...")
        fingerprint = self.generate_fingerprint()
        salt = get_random_bytes(16)
        encrypted_fingerprint = self.encrypt_fingerprint(fingerprint, salt)

        y_channel = ycbcr[:, :, 0]
        height, width = y_channel.shape
        fingerprint_flat = fingerprint.flatten()

        def process_block(block_idx):
            i, j = block_idx
            block = y_channel[i:i + self.block_size, j:j + self.block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            bit_idx = (i // self.block_size * (width // self.block_size) + j // self.block_size) % len(fingerprint_flat)
            embedded_block = self.embed_in_dct_block(dct_block, fingerprint_flat[bit_idx])
            return i, j, idct(idct(embedded_block.T, norm='ortho').T, norm='ortho')

        self.logger.info("Embedding fingerprint in image blocks...")
        block_indices = [(i, j) for i in range(0, height - self.block_size + 1, self.block_size)
                         for j in range(0, width - self.block_size + 1, self.block_size)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(process_block, block_indices), total=len(block_indices), desc="Embedding Progress"))

        self.logger.info("Applying embedded blocks to image...")
        for i, j, processed_block in results:
            y_channel[i:i + self.block_size, j:j + self.block_size] = processed_block

        ycbcr[:, :, 0] = y_channel
        rgb_image = self.ycbcr_to_rgb(ycbcr).astype(np.uint8)

        self.logger.info(f"Saving fingerprinted image to {output_path}...")
        Image.fromarray(rgb_image).save(output_path, format='PNG')

        self.logger.info("Fingerprint embedding complete.")
        return salt

    def verify_fingerprint(self, image_path, salt):
        self.logger.info("Starting fingerprint verification process...")
        img = np.array(Image.open(image_path))
        ycbcr = self.rgb_to_ycbcr(img)

        y_channel = ycbcr[:, :, 0]
        height, width = y_channel.shape
        extracted_fingerprint = np.zeros(self.fingerprint_size[0] * self.fingerprint_size[1] * 3, dtype=np.uint8)

        def process_block(block_idx):
            i, j = block_idx
            block = y_channel[i:i + self.block_size, j:j + self.block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            bits = self.extract_from_dct_block(dct_block)
            bit_idx = (i // self.block_size * (width // self.block_size) + j // self.block_size) % len(
                extracted_fingerprint)
            return bit_idx, bits

        self.logger.info("Extracting fingerprint from image blocks...")
        block_indices = [(i, j) for i in range(0, height - self.block_size + 1, self.block_size)
                         for j in range(0, width - self.block_size + 1, self.block_size)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(process_block, block_indices), total=len(block_indices), desc="Extraction Progress"))

        self.logger.info("Reconstructing extracted fingerprint...")
        for bit_idx, bits in results:
            extracted_fingerprint[bit_idx] = np.mean(bits)

        self.logger.debug("Extracted fingerprint sample:")
        self.logger.debug(extracted_fingerprint[:15])

        self.logger.info("Decoding extracted fingerprint...")
        decoded_fingerprint = np.zeros(self.fingerprint_size[0] * self.fingerprint_size[1], dtype=np.uint8)
        for i in range(0, len(extracted_fingerprint), 3):
            decoded_bit = np.mean(extracted_fingerprint[i:i + 3]) > 0.5
            decoded_fingerprint[i // 3] = decoded_bit

        self.logger.debug("Decoded fingerprint sample:")
        self.logger.debug(decoded_fingerprint[:15])

        self.logger.info("Generating original fingerprint for comparison...")
        original_fingerprint = self.generate_fingerprint()

        self.logger.info("Comparing fingerprints...")
        similarity = np.mean(original_fingerprint[::3] == decoded_fingerprint)
        self.logger.info(f"Fingerprint similarity: {similarity:.2%}")

        is_authentic = similarity > self.adaptive_threshold(image_path)
        self.logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")

        return is_authentic, similarity

    def adaptive_threshold(self, image_path):
        """Calculate adaptive threshold based on image quality."""
        psnr = self.calculate_psnr(image_path)
        base_threshold = 0.80
        max_threshold = 0.95
        min_threshold = 0.70
        quality_factor = min(max(psnr / 40, 0), 1)  # Assuming PSNR of 40 dB as high quality
        threshold = base_threshold + (max_threshold - base_threshold) * quality_factor
        return max(min(threshold, max_threshold), min_threshold)

    def calculate_psnr(self, image_path):
        """Calculate PSNR of the image."""
        img = cv2.imread(image_path)
        noise = np.ones_like(img) * 128
        return cv2.PSNR(img, noise)

    def assess_image_quality(self, original_path, fingerprinted_path):
        """Assess the quality of the fingerprinted image compared to the original."""
        original = cv2.imread(original_path)
        fingerprinted = cv2.imread(fingerprinted_path)

        psnr = cv2.PSNR(original, fingerprinted)
        ssim_value = ssim(original, fingerprinted, multichannel=True)

        return psnr, ssim_value

    def embed_fingerprint_with_quality(self, image_path, output_path):
        salt = self.embed_fingerprint(image_path, output_path)
        psnr, ssim_value = self.assess_image_quality(image_path, output_path)
        self.logger.info(f"Image Quality - PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}")
        return salt, psnr, ssim_value


# Additional utility functions for robustness testing
def apply_transformations(image_path, output_dir):
    """Apply various transformations to an image for robustness testing."""
    img = cv2.imread(image_path)

    # Rotation
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_dir, 'rotated.png'), rotated)

    # Scaling
    scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imwrite(os.path.join(output_dir, 'scaled.png'), scaled)

    # Cropping
    height, width = img.shape[:2]
    cropped = img[height // 4:3 * height // 4, width // 4:3 * width // 4]
    cv2.imwrite(os.path.join(output_dir, 'cropped.png'), cropped)

    # Compression
    cv2.imwrite(os.path.join(output_dir, 'compressed.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 50])


def batch_test(alphapunch, input_dir, output_dir):
    """Perform batch testing on multiple images."""
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)

            # Embed fingerprint
            salt = alphapunch.embed_fingerprint(image_path, os.path.join(output_dir, f'fp_{filename}'))

            # Apply transformations
            apply_transformations(os.path.join(output_dir, f'fp_{filename}'), output_dir)

            # Verify original and transformed images
            original_result, original_similarity = alphapunch.verify_fingerprint(
                os.path.join(output_dir, f'fp_{filename}'), salt)
            rotated_result, rotated_similarity = alphapunch.verify_fingerprint(os.path.join(output_dir, 'rotated.png'),
                                                                               salt)
            scaled_result, scaled_similarity = alphapunch.verify_fingerprint(os.path.join(output_dir, 'scaled.png'),
                                                                             salt)


            cropped_result, cropped_similarity = alphapunch.verify_fingerprint(os.path.join(output_dir, 'cropped.png'), salt)
            compressed_result, compressed_similarity = alphapunch.verify_fingerprint(os.path.join(output_dir, 'compressed.jpg'),
                                                                                     salt)

            results.append({
                'filename': filename,
                'original': (original_result, original_similarity),
                'rotated': (rotated_result, rotated_similarity),
                'scaled': (scaled_result, scaled_similarity),
                'cropped': (cropped_result, cropped_similarity),
                'compressed': (compressed_result, compressed_similarity)
            })

            return results


# Performance optimization function
def optimize_performance(func):
    """Decorator to optimize performance of a function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run.")
        return result

    return wrapper


# Apply the performance optimization decorator to the main methods
EnhancedAlphaPunch.embed_fingerprint = optimize_performance(EnhancedAlphaPunch.embed_fingerprint)
EnhancedAlphaPunch.verify_fingerprint = optimize_performance(EnhancedAlphaPunch.verify_fingerprint)

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('AlphaPunch')

    alphapunch = EnhancedAlphaPunch(private_key="your_secret_key_here", logger=logger)

    # Example usage
    salt, psnr, ssim = alphapunch.embed_fingerprint_with_quality('input_image.png', 'fingerprinted_image.png')
    print(f"Embedding complete. PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    is_authentic, similarity = alphapunch.verify_fingerprint('fingerprinted_image.png', salt)
    print(f"Verification result: {'Authentic' if is_authentic else 'Not authentic'}, Similarity: {similarity:.2%}")

    # Batch testing
    results = batch_test(alphapunch, 'input_images', 'output_images')
    for result in results:
        print(f"\nResults for {result['filename']}:")
        print(
            f"  Original: {'Authentic' if result['original'][0] else 'Not authentic'} (Similarity: {result['original'][1]:.2%})")
        print(
            f"  Rotated: {'Authentic' if result['rotated'][0] else 'Not authentic'} (Similarity: {result['rotated'][1]:.2%})")
        print(
            f"  Scaled: {'Authentic' if result['scaled'][0] else 'Not authentic'} (Similarity: {result['scaled'][1]:.2%})")
        print(
            f"  Cropped: {'Authentic' if result['cropped'][0] else 'Not authentic'} (Similarity: {result['cropped'][1]:.2%})")
        print(
            f"  Compressed: {'Authentic' if result['compressed'][0] else 'Not authentic'} (Similarity: {result['compressed'][1]:.2%})")

