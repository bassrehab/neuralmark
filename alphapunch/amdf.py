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


class AdaptiveMultiDomainFingerprinting:
    def __init__(self, fingerprint_size=(64, 64), embed_strength=0.15, tile_size=8):
        self.fingerprint_size = fingerprint_size
        self.embed_strength = embed_strength
        self.tile_size = tile_size
        self.feature_extractor = self._build_feature_extractor()
        self.fingerprint_generator = self._build_fingerprint_generator()
        self.verifier = self._build_verifier()

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
        img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        img_tensor = tf.image.resize(img_tensor, (224, 224))
        img_tensor = preprocess_input(img_tensor)
        features = self.feature_extractor(tf.expand_dims(img_tensor, 0))
        fingerprint = self.fingerprint_generator(features)
        return tf.image.resize(fingerprint, self.fingerprint_size).numpy().squeeze()

    def embed_fingerprint(self, image, fingerprint):
        # Convert image to float32
        image = image.astype(np.float32)
        original_shape = image.shape

        # Pad the image if dimensions are not even
        pad_h = 0 if image.shape[0] % 2 == 0 else 1
        pad_w = 0 if image.shape[1] % 2 == 0 else 1
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        # Apply DWT to each color channel
        coeffs = [pywt.dwt2(image[:, :, i], 'haar') for i in range(3)]

        # Tile the fingerprint to match the size of the approximation coefficients
        h, w = coeffs[0][0].shape
        tiled_fingerprint = np.tile(fingerprint, (h // fingerprint.shape[0] + 1, w // fingerprint.shape[1] + 1))
        tiled_fingerprint = tiled_fingerprint[:h, :w]

        # Embed the fingerprint in the approximation coefficients of each channel
        for i in range(3):
            cA, (cH, cV, cD) = coeffs[i]
            cA += self.embed_strength * tiled_fingerprint
            coeffs[i] = (cA, (cH, cV, cD))

        # Apply inverse DWT to each channel
        embedded_image = np.stack([pywt.idwt2(coeff, 'haar') for coeff in coeffs], axis=-1)

        # Remove padding if added
        embedded_image = embedded_image[:original_shape[0], :original_shape[1], :]

        return np.clip(embedded_image, 0, 255).astype(np.uint8)

    def extract_fingerprint(self, image):
        # Convert image to float32
        image = image.astype(np.float32)

        # Pad the image if dimensions are not even
        pad_h = 0 if image.shape[0] % 2 == 0 else 1
        pad_w = 0 if image.shape[1] % 2 == 0 else 1
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        # Apply DWT to each color channel
        coeffs = [pywt.dwt2(image[:, :, i], 'haar') for i in range(3)]

        # Extract fingerprint from the approximation coefficients of the first channel
        cA, _ = coeffs[0]
        extracted = cA / self.embed_strength

        # Reshape to original fingerprint size
        h, w = self.fingerprint_size
        return cv2.resize(extracted, (w, h))

    def apply_error_correction(self, fingerprint):
        # Simple error correction: apply median filter
        return cv2.medianBlur(fingerprint.astype(np.float32), 3)

    def verify_fingerprint(self, image, original_fingerprint):
        extracted = self.extract_fingerprint(image)
        extracted = self.apply_error_correction(extracted)
        difference = np.abs(extracted - original_fingerprint)
        return self.verifier.predict(difference[np.newaxis, ..., np.newaxis])[0, 0]

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
