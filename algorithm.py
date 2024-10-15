import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import hashlib
import concurrent.futures
from tqdm import tqdm
import logging

class ImprovedAlphaPunch:
    def __init__(self, private_key, logger, fingerprint_size=(64, 64), block_size=8, embed_strength=0.75):
        self.private_key = private_key.encode()
        self.fingerprint_size = fingerprint_size
        self.block_size = block_size
        self.embed_positions = [(block_size // 2, block_size // 2 - 1), (block_size // 2 - 1, block_size // 2)]
        self.logger = logger
        self.embed_strength = embed_strength

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
        cipher = AES.new(key, AES.MODE_ECB)
        flattened = fingerprint.tobytes()
        padded = pad(flattened, AES.block_size)
        encrypted = cipher.encrypt(padded)
        self.logger.debug(f"Encrypted fingerprint length: {len(encrypted)}")
        return encrypted

    def decrypt_fingerprint(self, encrypted_fingerprint, salt):
        key = hashlib.sha256(self.private_key + salt).digest()
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted = cipher.decrypt(encrypted_fingerprint)
        try:
            unpadded = unpad(decrypted, AES.block_size)
        except ValueError:
            self.logger.warning("Padding error during decryption. Using raw decrypted data.")
            unpadded = decrypted
        decrypted_fingerprint = np.frombuffer(unpadded, dtype=np.uint8).reshape(-1)
        self.logger.debug(f"Decrypted fingerprint shape: {decrypted_fingerprint.shape}")
        self.logger.debug(f"Decrypted fingerprint sample: {decrypted_fingerprint[:15]}")
        return decrypted_fingerprint

    def rgb_to_ycbcr(self, rgb):
        rgb = rgb.astype(np.float32) / 255.0
        transform = np.array([[0.299, 0.587, 0.114],
                              [-0.168736, -0.331264, 0.5],
                              [0.5, -0.418688, -0.081312]])
        ycbcr = rgb.dot(transform.T)
        ycbcr[:, :, 1:] += 0.5
        return np.clip(ycbcr, 0, 1) * 255

    def ycbcr_to_rgb(self, ycbcr):
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
        similarity = np.mean(original_fingerprint == decoded_fingerprint)
        self.logger.info(f"Fingerprint similarity: {similarity:.2%}")

        is_authentic = similarity > 0.80  # Lowered threshold
        self.logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")

        return is_authentic


if __name__ == "__main__":
    logger = logging.getLogger('AlphaPunch')

    logger.info("Initializing AlphaPunch...")
    alphapunch = ImprovedAlphaPunch(private_key="your_secret_key_here", logger=logger)

    logger.info("Embedding fingerprint...")
    salt = alphapunch.embed_fingerprint("input_image.jpg", "fingerprinted_image.jpg")

    logger.info("Verifying fingerprint...")
    is_authentic = alphapunch.verify_fingerprint("fingerprinted_image.jpg", salt)

    logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")
    logger.info("Process complete.")