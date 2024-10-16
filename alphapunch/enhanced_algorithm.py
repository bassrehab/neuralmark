import concurrent.futures
import hashlib
import time
import traceback

import cv2
import numpy as np
import pywt
import tensorflow as tf
from PIL import Image
from phe import paillier
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class EnhancedAlphaPunch:
    def __init__(self, private_key, logger, fingerprint_size=(64, 64), block_size=8, embed_strength=0.75):
        self.private_key = private_key.encode()
        self.fingerprint_size = fingerprint_size
        self.block_size = block_size
        self.embed_positions = [(block_size // 2, block_size // 2 - 1), (block_size // 2 - 1, block_size // 2)]
        self.logger = logger
        self.embed_strength = embed_strength
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def box_counting_dimension(self, image, max_box_size=None, min_box_size=2):
        if len(image.shape) == 3:
            img = np.mean(image, axis=2).astype(np.uint8)
        else:
            img = image.astype(np.uint8)

        if max_box_size is None:
            max_box_size = min(img.shape)

        # Ensure we have at least 2 different box sizes
        if max_box_size <= min_box_size:
            return 0

        img = (img > img.mean()).astype(np.uint8)

        box_sizes = np.floor(np.logspace(np.log2(min_box_size), np.log2(max_box_size), num=10, base=2)).astype(int)
        box_sizes = np.unique(box_sizes)

        counts = []
        for size in box_sizes:
            padded_size = (np.ceil(np.array(img.shape) / size) * size).astype(int)
            padded_img = np.zeros(padded_size)
            padded_img[:img.shape[0], :img.shape[1]] = img

            box_count = (padded_img.reshape(padded_size[0] // size, size, -1, size)
                         .sum(axis=(1, 3)) > 0).sum()
            counts.append(max(1, box_count))

        valid_indices = np.array(counts) > 0
        if np.sum(valid_indices) < 2:
            return 0

        coeffs = np.polyfit(np.log(box_sizes[valid_indices]), np.log(np.array(counts)[valid_indices]), 1)
        return -coeffs[0]

    def generate_fractal_fingerprint(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate the step size to ensure we get the correct number of fractal dimensions
        steps = (self.fingerprint_size[0] * self.fingerprint_size[1])
        step_y = max(1, gray_image.shape[0] // self.fingerprint_size[0])
        step_x = max(1, gray_image.shape[1] // self.fingerprint_size[1])

        fractal_dims = []
        for i in range(0, gray_image.shape[0], step_y):
            for j in range(0, gray_image.shape[1], step_x):
                region = gray_image[i:i + step_y, j:j + step_x]
                fd = self.box_counting_dimension(region)
                fractal_dims.append(fd)

                if len(fractal_dims) == steps:
                    break
            if len(fractal_dims) == steps:
                break

        # If we don't have enough dimensions, pad with zeros
        fractal_dims += [0] * (steps - len(fractal_dims))

        # If we have too many dimensions, truncate
        fractal_dims = fractal_dims[:steps]

        median_fd = np.median(fractal_dims)
        fingerprint = (np.array(fractal_dims) > median_fd).astype(np.uint8)
        return fingerprint.reshape(self.fingerprint_size)

    def add_error_correction(self, fingerprint):
        # Simple repetition code
        return np.repeat(np.repeat(fingerprint, 3, axis=0), 3, axis=1)

    def remove_error_correction(self, extracted_fingerprint):
        # Majority voting
        corrected = (convolve2d(extracted_fingerprint, np.ones((3, 3)), mode='valid') > 4).astype(int)
        # Resize to match the original fingerprint size
        return cv2.resize(corrected, self.fingerprint_size, interpolation=cv2.INTER_NEAREST)

    def quantum_inspired_embed(self, dct_block, bit):
        embed_strength = 0.1  # Increase this value for stronger embedding
        for y, x in self.embed_positions:
            if bit:
                dct_block[y, x] += embed_strength
            else:
                dct_block[y, x] -= embed_strength
        return dct_block

    def wavelet_embed(self, image, fingerprint):
        # Convert image to float32 and scale to [0, 1]
        img_float = image.astype(np.float32) / 255.0

        # Apply wavelet embedding to each color channel
        embedded_channels = []
        for channel in range(3):
            coeffs = pywt.wavedec2(img_float[:, :, channel], 'db1', level=3)

            fingerprint_flat = fingerprint.flatten()
            coeff_index = 0

            for i in range(1, 3):
                for j in range(3):
                    coeff = coeffs[i][j]
                    embed_size = min(coeff.size, fingerprint_flat.size - coeff_index)
                    coeff_flat = coeff.flatten()
                    coeff_flat[:embed_size] += self.embed_strength * (
                                2 * fingerprint_flat[coeff_index:coeff_index + embed_size] - 1)
                    coeffs[i] = (coeffs[i][0], coeffs[i][1], coeffs[i][2])
                    coeff_index += embed_size
                    if coeff_index >= fingerprint_flat.size:
                        break
                if coeff_index >= fingerprint_flat.size:
                    break

            embedded_channels.append(pywt.waverec2(coeffs, 'db1'))

        # Combine channels and clip to [0, 1]
        embedded_img = np.clip(np.stack(embedded_channels, axis=-1), 0, 1)

        # Convert back to uint8
        return (embedded_img * 255).astype(np.uint8)

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
        embedded_image = image.copy()
        fingerprint_flat = fingerprint.flatten()
        for i, bit in enumerate(fingerprint_flat):
            if i < image.shape[0] * image.shape[1]:
                y = i // image.shape[1]
                x = i % image.shape[1]
                channel = i % 3
                embedded_image[y, x, channel] = (embedded_image[y, x, channel] & 0xFE) | bit
        return embedded_image

    def embed_fingerprint(self, image_path, output_path):
        self.logger.info("Starting enhanced fingerprint embedding process...")
        start_time = time.time()

        try:
            img = np.array(Image.open(image_path))
            self.logger.info(f"Image loaded. Shape: {img.shape}. Time: {time.time() - start_time:.2f}s")

            fingerprint = self.generate_fractal_fingerprint(img)
            self.logger.info(f"Fractal fingerprint generated. Time: {time.time() - start_time:.2f}s")

            # Add error correction
            fingerprint_with_ec = self.add_error_correction(fingerprint)
            self.logger.info(f"Error correction added to fingerprint. New shape: {fingerprint_with_ec.shape}")

            embedded_img = self.wavelet_embed(img, fingerprint_with_ec)
            self.logger.info(f"Wavelet embedding completed. Time: {time.time() - start_time:.2f}s")

            def process_block(block_idx):
                i, j = block_idx
                block = embedded_img[i:i + self.block_size, j:j + self.block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                bit_idx = (i // self.block_size * (
                        embedded_img.shape[1] // self.block_size) + j // self.block_size) % fingerprint.size
                embedded_block = self.quantum_inspired_embed(dct_block, fingerprint.flat[bit_idx])
                return i, j, idct(idct(embedded_block.T, norm='ortho').T, norm='ortho')

            block_indices = [(i, j) for i in range(0, embedded_img.shape[0] - self.block_size + 1, self.block_size)
                             for j in range(0, embedded_img.shape[1] - self.block_size + 1, self.block_size)]

            self.logger.info(
                f"Starting DCT embedding. Total blocks: {len(block_indices)}. Time: {time.time() - start_time:.2f}s")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(process_block, block_indices), total=len(block_indices),
                                    desc="Embedding Progress"))

            self.logger.info(f"DCT embedding completed. Time: {time.time() - start_time:.2f}s")

            for i, j, block in results:
                embedded_img[i:i + self.block_size, j:j + self.block_size] = block

            self.blockchain_fingerprint(embedded_img)
            self.logger.info(f"Blockchain-inspired chaining completed. Time: {time.time() - start_time:.2f}s")

            encrypted_img = self.homomorphic_embed(embedded_img, fingerprint)
            self.logger.info(f"Homomorphic encryption completed. Time: {time.time() - start_time:.2f}s")

            Image.fromarray(encrypted_img).save(output_path)
            self.logger.info(f"Fingerprinted image saved. Total time: {time.time() - start_time:.2f}s")

            # Calculate PSNR and SSIM with adjusted parameters
            psnr = peak_signal_noise_ratio(img, encrypted_img)
            win_size = min(7,
                           min(img.shape[0], img.shape[1]) - 1)  # Ensure window size is odd and not larger than image
            if win_size % 2 == 0:
                win_size -= 1
            ssim = structural_similarity(img, encrypted_img, win_size=win_size, channel_axis=2, data_range=255)
            self.logger.info(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

            return fingerprint, psnr, ssim  # Return fingerprint without error correction

        except Exception as e:
            self.logger.error(f"Error in embed_fingerprint: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def verify_fingerprint(self, image_path, original_fingerprint):
        self.logger.info("Starting fingerprint verification process...")
        start_time = time.time()
        img = np.array(Image.open(image_path))

        # Extract the embedded fingerprint
        extracted_fingerprint = self.extract_fingerprint(img)
        self.logger.info(f"Extracted fingerprint. Time: {time.time() - start_time:.2f}s")

        # Remove error correction
        extracted_fingerprint_no_ec = self.remove_error_correction(extracted_fingerprint)
        self.logger.info(
            f"Error correction removed from extracted fingerprint. Shape: {extracted_fingerprint_no_ec.shape}")

        # Ensure shapes match
        if extracted_fingerprint_no_ec.shape != original_fingerprint.shape:
            self.logger.warning(
                f"Shape mismatch: extracted {extracted_fingerprint_no_ec.shape} vs original {original_fingerprint.shape}")
            extracted_fingerprint_no_ec = cv2.resize(extracted_fingerprint_no_ec,
                                                     (original_fingerprint.shape[1], original_fingerprint.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)

        # Compare with original fingerprint
        similarity = np.mean(extracted_fingerprint_no_ec == original_fingerprint)
        self.logger.info(f"Fingerprint similarity: {similarity:.2%}")

        # Use a more robust comparison
        hamming_distance = np.sum(extracted_fingerprint_no_ec != original_fingerprint)
        normalized_hamming_distance = hamming_distance / (original_fingerprint.shape[0] * original_fingerprint.shape[1])
        self.logger.info(f"Normalized Hamming distance: {normalized_hamming_distance:.2%}")

        is_authentic = normalized_hamming_distance < 0.3  # Adjust threshold as needed
        self.logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")

        self.logger.info(f"Verification completed. Total time: {time.time() - start_time:.2f}s")
        return is_authentic, similarity, normalized_hamming_distance

    def extract_fingerprint(self, img):
        fingerprint_size_with_ec = (self.fingerprint_size[0] * 3, self.fingerprint_size[1] * 3)
        extracted_bits = np.zeros(fingerprint_size_with_ec, dtype=int)

        for y in range(fingerprint_size_with_ec[0]):
            for x in range(fingerprint_size_with_ec[1]):
                pixel_index = (y * img.shape[1] + x) % (img.shape[0] * img.shape[1])
                channel = pixel_index % 3
                extracted_bits[y, x] = img.flat[pixel_index * 3 + channel] & 0x01

        return extracted_bits

    def extract_bit_from_dct(self, dct_block):
        # Extract the bit based on the embedding positions
        values = [dct_block[y, x] for y, x in self.embed_positions]
        return int(np.mean(values) > 0)

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
        fingerprint, psnr, ssim = self.embed_fingerprint(image_path, output_path)
        self.logger.info(f"Image Quality - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        return fingerprint, psnr, ssim


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('EnhancedAlphaPunch')

    alphapunch = EnhancedAlphaPunch(private_key="your_secret_key_here", logger=logger)

    # Example usage
    fingerprint, psnr, ssim = alphapunch.embed_fingerprint_with_quality('input_image.jpg', 'fingerprinted_image.png')
    print(f"Embedding complete. PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    is_authentic, similarity, normalized_hamming_distance = alphapunch.verify_fingerprint('fingerprinted_image.png', fingerprint)
    print(f"Verification result: {'Authentic' if is_authentic else 'Not authentic'}, Similarity: {similarity:.2%}, Normalized Hamming Distance: {normalized_hamming_distance:.2%}")
