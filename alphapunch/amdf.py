# amdf.py

import numpy as np
import cv2
import tensorflow as tf
from scipy.fftpack import dct, idct
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from scipy.signal import convolve2d
import pywt
from skimage.metrics import structural_similarity


class AdaptiveMultiDomainFingerprinting:
    def __init__(self, fingerprint_size=(64, 64), embed_strength=0.4, config=None):
        self.fingerprint_size = fingerprint_size
        self.embed_strength = embed_strength
        self.config = config  # Store the config

        # Store config sections for easier access
        if config:
            self.verification_config = config['algorithm']['verification']
            self.error_correction_config = config['algorithm']['error_correction']
        else:
            # Default configurations if no config provided
            self.verification_config = {
                'ncc_weight': 0.5,
                'ssim_weight': 0.3,
                'min_individual_score': 0.4
            }
            self.error_correction_config = {
                'gaussian_kernel_size': 3,
                'gaussian_sigma': 0.5
            }

        # Initialize models
        self.feature_extractor = self._build_feature_extractor()
        self.fingerprint_generator = self._build_fingerprint_generator()
        self.verifier = self._build_verifier()

        # Set random seeds if config is provided
        if config and 'testing' in config:
            np.random.seed(config['testing']['random_seed'])
            tf.random.set_seed(config['testing']['random_seed'])

    def _build_feature_extractor(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        return Model(inputs=base_model.input, outputs=base_model.get_layer('block3_conv3').output)

    def _build_fingerprint_generator(self):
        input_shape = self.feature_extractor.output_shape[1:]
        inputs = Input(shape=input_shape)

        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Decoder
        conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
        up1 = UpSampling2D(size=(2, 2))(conv3)
        conv4 = Conv2D(64, 3, activation='relu', padding='same')(up1)
        up2 = UpSampling2D(size=(2, 2))(conv4)
        outputs = Conv2D(1, 3, activation='sigmoid', padding='same')(up2)

        return Model(inputs=inputs, outputs=outputs)

    def _build_verifier(self):
        inputs = Input(shape=self.fingerprint_size + (1,))
        x = Conv2D(32, 3, activation='relu')(inputs)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, 3, activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def generate_fingerprint(self, image):
        # Preprocess image for VGG16
        img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        img_tensor = tf.image.resize(img_tensor, (224, 224))
        img_tensor = preprocess_input(img_tensor)

        # Extract features and generate fingerprint
        features = self.feature_extractor(tf.expand_dims(img_tensor, 0))
        fingerprint = self.fingerprint_generator(features)

        # Resize to desired fingerprint size
        return tf.image.resize(fingerprint, self.fingerprint_size).numpy().squeeze()

    def embed_fingerprint(self, image, fingerprint):
        image = image.astype(np.float32)
        original_shape = image.shape

        # Pre-process fingerprint
        fingerprint = cv2.normalize(fingerprint, None, 0, 1, cv2.NORM_MINMAX)
        fingerprint = cv2.GaussianBlur(fingerprint, (3, 3), 0.5)

        # Decompose image into multiple levels
        coeffs = [pywt.wavedec2(image[:, :, i], 'haar', level=3) for i in range(3)]

        for i in range(3):
            # Get all coefficients
            c = list(coeffs[i])

            # Embed in approximation coefficients (strongest)
            h, w = c[0].shape
            tiled = cv2.resize(fingerprint, (w, h))
            c[0] += self.embed_strength * 2.0 * tiled

            # Embed in first level details (medium strength)
            h, w = c[1][0].shape
            tiled = cv2.resize(fingerprint, (w, h))
            c[1] = tuple(coef + self.embed_strength * 0.5 * tiled for coef in c[1])

            # Embed in second level details (weak strength)
            h, w = c[2][0].shape
            tiled = cv2.resize(fingerprint, (w, h))
            c[2] = tuple(coef + self.embed_strength * 0.25 * tiled for coef in c[2])

            coeffs[i] = tuple(c)

        # Reconstruct image
        embedded_image = np.stack([pywt.waverec2(coeff, 'haar') for coeff in coeffs], axis=-1)
        embedded_image = embedded_image[:original_shape[0], :original_shape[1], :]

        return np.clip(embedded_image, 0, 255).astype(np.uint8)

    def extract_fingerprint(self, image):
        image = image.astype(np.float32)

        # Extract from multiple wavelet levels
        fingerprints = []

        for i in range(3):
            coeffs = pywt.wavedec2(image[:, :, i], 'haar', level=3)

            # Extract from each level with appropriate weights
            f1 = coeffs[0] / (self.embed_strength * 2.0)
            f2 = coeffs[1][0] / (self.embed_strength * 0.5)
            f3 = coeffs[2][0] / (self.embed_strength * 0.25)

            # Resize all to fingerprint size
            h, w = self.fingerprint_size
            f1 = cv2.resize(f1, (w, h))
            f2 = cv2.resize(f2, (w, h))
            f3 = cv2.resize(f3, (w, h))

            # Weighted combination
            fingerprint = 0.6 * f1 + 0.3 * f2 + 0.1 * f3
            fingerprints.append(fingerprint)

        # Combine fingerprints from all channels
        extracted = np.mean(fingerprints, axis=0)
        extracted = cv2.normalize(extracted, None, 0, 1, cv2.NORM_MINMAX)

        return extracted

    def apply_error_correction(self, fingerprint):
        """Apply error correction to the fingerprint."""
        # Ensure kernel size is odd
        kernel_size = self.error_correction_config['gaussian_kernel_size']
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = self.error_correction_config['gaussian_sigma']

        # Apply Gaussian blur for denoising
        denoised = cv2.GaussianBlur(
            fingerprint,
            (kernel_size, kernel_size),
            sigma
        )

        # Normalize output
        corrected = cv2.normalize(denoised, None, 0, 1, cv2.NORM_MINMAX)

        return corrected

    def verify_fingerprint(self, image, original_fingerprint):
        extracted = self.extract_fingerprint(image)
        extracted = self.apply_error_correction(extracted)

        # Normalize fingerprints
        extracted = cv2.normalize(extracted, None, 0, 1, cv2.NORM_MINMAX)
        original = cv2.normalize(original_fingerprint, None, 0, 1, cv2.NORM_MINMAX)

        ver_config = self.config['algorithm']['verification']

        # Calculate multiple similarity metrics
        ncc = cv2.matchTemplate(extracted, original, cv2.TM_CCORR_NORMED)[0][0]
        ssim = structural_similarity(extracted, original, data_range=1.0)

        # Combined similarity with configured weights
        similarity = (ncc * ver_config['ncc_weight'] +
                      ssim * ver_config['ssim_weight'])

        # Penalty for low individual scores
        min_score = ver_config['min_individual_score']
        if ncc < min_score or ssim < min_score:
            similarity *= 0.8

        return similarity

    def train_verifier(self, authentic_pairs, fake_pairs, epochs=20, batch_size=32, validation_split=0.2):
        X = []
        y = []
        for orig, embedded in authentic_pairs:
            diff = np.abs(self.extract_fingerprint(embedded) - self.generate_fingerprint(orig))
            X.append(diff)
            y.append(1)
        for orig, fake in fake_pairs:
            diff = np.abs(self.extract_fingerprint(fake) - self.generate_fingerprint(orig))
            X.append(diff)
            y.append(0)
        X = np.array(X)[..., np.newaxis]
        y = np.array(y)
        self.verifier.fit(X, y, epochs=epochs, validation_split=validation_split, batch_size=batch_size)
