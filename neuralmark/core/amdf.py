from typing import Tuple, List

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.models import Model

import cv2
import numpy as np
import pywt
import tensorflow as tf
from scipy import stats
from skimage.metrics import structural_similarity


class AdaptiveMultiDomainFingerprinting:
    def __init__(self, config=None, logger=None):
        """Initialize AMDF with configuration."""
        self.config = config
        self.logger = logger

        # Initialize parameters from config
        if config and 'algorithm' in config:
            self.fingerprint_size = tuple(config['algorithm']['fingerprint_size'])
            self.embed_strength = config['algorithm']['embed_strength']
            self.verification_weights = config.get('verification', {}).get('stage_weights',
                                                                           [0.3, 0.3, 0.2, 0.2])
            self.feature_weights = config.get('algorithm', {}).get('feature_weights', {
                'vgg': 0.4, 'wavelet': 0.3, 'dct': 0.3
            })
        else:
            self.fingerprint_size = (64, 64)
            self.embed_strength = 0.4
            self.verification_weights = [0.3, 0.3, 0.2, 0.2]
            self.feature_weights = {'vgg': 0.4, 'wavelet': 0.3, 'dct': 0.3}

        # Initialize wavelet parameters
        self.wavelet = 'db1'  # Daubechies wavelet
        self.wavelet_level = 3  # Decomposition level

        # Set random seeds if config is provided
        if config and 'testing' in config:
            np.random.seed(config['testing']['random_seed'])
            tf.random.set_seed(config['testing']['random_seed'])

        # Initialize models
        self.feature_extractor = self._build_feature_extractor()
        self.fingerprint_generator = self._build_fingerprint_generator()
        self.verifier = self._build_verifier()

        if self.logger:
            self.logger.debug("AdaptiveMultiDomainFingerprinting initialized successfully")

    def _build_feature_extractor(self):
        """Build VGG-based feature extractor."""
        import tensorflow as tf

        with tf.device('/CPU:0'):
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.feature_model = Model(inputs=base_model.input,
                                       outputs=base_model.get_layer('block3_conv3').output)

            # Return a wrapped function that handles tensor formatting
            def feature_extractor(inputs):
                inputs = tf.cast(inputs, tf.float32)
                if len(tf.shape(inputs)) == 3:
                    inputs = tf.expand_dims(inputs, 0)
                return self.feature_model(inputs)

            return feature_extractor

    def _build_fingerprint_generator(self):
        """Build fingerprint generator network."""
        # Use the feature model's output shape instead of the wrapper function
        input_shape = self.feature_model.output_shape[1:]
        inputs = Input(shape=input_shape)

        # Encoder
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        # Decoder
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        return Model(inputs=inputs, outputs=outputs)

    def _extract_wavelet_features(self, image: np.ndarray) -> np.ndarray:
        """Extract wavelet features from image."""
        try:
            # Ensure image is 2D
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply wavelet decomposition
            coeffs = pywt.wavedec2(image, self.wavelet, level=self.wavelet_level)

            # Extract features from coefficients
            features = []
            for level in coeffs[1:]:  # Skip approximation coefficients
                for detail in level:
                    # Calculate statistical features
                    features.extend([
                        np.mean(np.abs(detail)),
                        np.std(detail),
                        stats.skew(detail.flatten()),
                        stats.kurtosis(detail.flatten())
                    ])

            # Normalize features
            features = np.array(features)
            if np.any(features):  # Check for non-zero features
                features = features / (np.linalg.norm(features) + 1e-10)

            return features

        except Exception as e:
            self.logger.error(f"Error in wavelet feature extraction: {str(e)}")
            raise

    def _extract_multi_scale_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features at multiple scales with fixed shapes."""
        # Use single scale for stability
        scales = [1.0]
        features = []

        try:
            # Convert to RGB once
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Pre-compute grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            for scale in scales:
                try:
                    # Prepare VGG input
                    vgg_input = cv2.resize(image, (224, 224))
                    vgg_input = vgg_input.astype(np.float32)
                    vgg_input = np.expand_dims(vgg_input, 0)
                    vgg_input = preprocess_input(vgg_input)

                    # Extract VGG features
                    vgg_features = self.feature_extractor(vgg_input)
                    vgg_features = tf.reduce_mean(vgg_features, axis=[1, 2]).numpy().flatten()

                    # Extract wavelet features
                    wavelet_features = self._extract_wavelet_features(gray)

                    # Extract DCT features
                    dct_features = self._extract_dct_features(gray)

                    # Combine features with weights
                    combined = np.concatenate([
                        self.feature_weights['vgg'] * vgg_features,
                        self.feature_weights['wavelet'] * wavelet_features,
                        self.feature_weights['dct'] * dct_features
                    ])
                    features.append(combined)

                except Exception as e:
                    self.logger.error(f"Error processing scale {scale}: {str(e)}")
                    continue

            if not features:
                raise ValueError("Feature extraction failed for all scales")

            # Average features across scales
            return np.mean(features, axis=0)

        except Exception as e:
            self.logger.error(f"Error in feature extraction: {str(e)}")
            raise

    def _build_verifier(self):
        """Build verification network."""
        input_shape = self.fingerprint_size + (1,)
        inputs = Input(shape=input_shape)

        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Generate fingerprint using multi-scale features."""
        features = self._extract_multi_scale_features(image)
        fingerprint = self._convert_features_to_fingerprint(features)
        return self.apply_error_correction(fingerprint)

    def _process_wavelet_coeffs(self, coeffs) -> np.ndarray:
        """Process wavelet coefficients into features."""
        features = []
        for level in coeffs[1:]:
            for detail in level:
                features.extend([
                    np.mean(np.abs(detail)),
                    np.std(detail),
                    stats.skew(detail.flatten()),
                    stats.kurtosis(detail.flatten())
                ])
        return np.array(features)

    def _ensure_even_size(self, image: np.ndarray) -> np.ndarray:
        """Ensure image dimensions are even for DCT."""
        h, w = image.shape[:2]
        new_h = h - (h % 2)  # Make height even
        new_w = w - (w % 2)  # Make width even
        return image[:new_h, :new_w]

    def _extract_dct_features(self, image: np.ndarray) -> np.ndarray:
        """Extract DCT features ensuring even dimensions."""
        try:
            # Ensure image is grayscale
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Ensure dimensions are even
            image = self._ensure_even_size(image)

            # Apply DCT
            dct_coeffs = cv2.dct(np.float32(image))

            # Take top-left corner coefficients
            features = dct_coeffs[:8, :8].flatten()

            # Normalize features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features

        except Exception as e:
            self.logger.error(f"Error extracting DCT features: {str(e)}")
            # Return zero features in case of error
            return np.zeros(64)

    def _convert_features_to_fingerprint(self, features: np.ndarray) -> np.ndarray:
        """Convert feature vector to fingerprint matrix."""
        # Reshape features to match fingerprint size
        features = cv2.resize(features.reshape(-1, 1), self.fingerprint_size)
        return cv2.normalize(features, None, 0, 1, cv2.NORM_MINMAX)

    def apply_error_correction(self, fingerprint: np.ndarray) -> np.ndarray:
        """Apply error correction to fingerprint."""
        # Apply Gaussian smoothing for noise reduction
        kernel_size = self.config.get('algorithm', {}).get('gaussian_kernel_size', 3)
        sigma = self.config.get('algorithm', {}).get('gaussian_sigma', 0.5)

        smoothed = cv2.GaussianBlur(fingerprint, (kernel_size, kernel_size), sigma)
        return cv2.normalize(smoothed, None, 0, 1, cv2.NORM_MINMAX)

    def _calculate_embedding_mask(self, image: np.ndarray) -> np.ndarray:
        """Calculate adaptive embedding mask."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate edge map
        edges = cv2.Canny(gray, 100, 200)

        # Calculate texture map
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(texture)
        texture = texture / texture.max()

        # Combine masks
        mask = (edges / 255.0 + texture) / 2
        return cv2.GaussianBlur(mask, (5, 5), 0)

    def extract_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Extract fingerprint from image."""
        fingerprint = np.zeros(self.fingerprint_size)

        # Extract from each color channel
        for i in range(3):
            # Wavelet decomposition
            coeffs = pywt.wavedec2(image[:, :, i], 'db1', level=3)

            # Extract from approximation coefficients
            extracted = coeffs[0] / self.embed_strength

            # Resize to fingerprint size
            resized = cv2.resize(extracted, self.fingerprint_size)

            # Accumulate
            fingerprint += resized

        # Average and normalize
        fingerprint /= 3.0
        return cv2.normalize(fingerprint, None, 0, 1, cv2.NORM_MINMAX)

    def _statistical_verification(self, image: np.ndarray,
                                  original_fingerprint: np.ndarray) -> Tuple[float, List[str]]:
        """Perform statistical verification."""
        current_fp = self.extract_fingerprint(image)
        modifications = []

        # Calculate statistical metrics
        mean_diff = np.abs(np.mean(current_fp) - np.mean(original_fingerprint))
        std_diff = np.abs(np.std(current_fp) - np.std(original_fingerprint))

        # Histogram comparison
        hist1, _ = np.histogram(current_fp, bins=50, range=(0, 1))
        hist2, _ = np.histogram(original_fingerprint, bins=50, range=(0, 1))
        hist_similarity = np.sum(np.minimum(hist1, hist2)) / np.sum(hist1)

        if mean_diff > 0.2:
            modifications.append("Intensity modification")
        if std_diff > 0.2:
            modifications.append("Contrast modification")

        similarity = 1.0 - (mean_diff + std_diff) / 2
        similarity = similarity * 0.7 + hist_similarity * 0.3

        return similarity, modifications

    def _geometric_verification(self, image: np.ndarray,
                                original_fingerprint: np.ndarray) -> Tuple[float, List[str]]:
        """Perform geometric verification."""
        current_fp = self.extract_fingerprint(image)
        modifications = []

        # Calculate moments
        current_moments = cv2.moments(current_fp)
        original_moments = cv2.moments(original_fingerprint)

        moment_diffs = []
        for moment in ['mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03']:
            if original_moments['m00'] != 0 and current_moments['m00'] != 0:
                orig_norm = original_moments[moment] / (original_moments['m00'] ** 2)
                curr_norm = current_moments[moment] / (current_moments['m00'] ** 2)
                moment_diffs.append(abs(orig_norm - curr_norm))

        if moment_diffs:
            geometric_similarity = 1.0 - min(1.0, np.mean(moment_diffs) * 10)
            if geometric_similarity < 0.7:
                modifications.append("Geometric transformation")
        else:
            geometric_similarity = 0.0

        return geometric_similarity, modifications

    def _compare_wavelets(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compare fingerprints in wavelet domain."""
        try:
            # Ensure inputs are 2D arrays
            if len(fp1.shape) > 2:
                fp1 = cv2.cvtColor(fp1.astype(np.float32), cv2.COLOR_BGR2GRAY)
            if len(fp2.shape) > 2:
                fp2 = cv2.cvtColor(fp2.astype(np.float32), cv2.COLOR_BGR2GRAY)

            # Normalize inputs
            fp1 = cv2.normalize(fp1, None, 0, 1, cv2.NORM_MINMAX)
            fp2 = cv2.normalize(fp2, None, 0, 1, cv2.NORM_MINMAX)

            # Decompose both fingerprints
            coeffs1 = pywt.wavedec2(fp1, self.wavelet, level=self.wavelet_level)
            coeffs2 = pywt.wavedec2(fp2, self.wavelet, level=self.wavelet_level)

            # Compare coefficients
            similarities = []

            # Compare approximation coefficients
            sim = np.corrcoef(coeffs1[0].flatten(), coeffs2[0].flatten())[0, 1]
            similarities.append(sim)

            # Compare detail coefficients
            for c1, c2 in zip(coeffs1[1:], coeffs2[1:]):
                for d1, d2 in zip(c1, c2):
                    sim = np.corrcoef(d1.flatten(), d2.flatten())[0, 1]
                    similarities.append(sim)

            # Handle NaN values that might occur from correlation
            similarities = np.nan_to_num(similarities, nan=0.0)

            # Return weighted average of similarities
            weights = [2.0] + [1.0] * (len(similarities) - 1)  # Give more weight to approximation
            weighted_sim = np.average(similarities, weights=weights)

            return max(0.0, min(1.0, weighted_sim))  # Ensure result is between 0 and 1

        except Exception as e:
            self.logger.error(f"Error in _compare_wavelets: {str(e)}")
            if isinstance(e, ValueError):
                # Return a default similarity score in case of error
                return 0.0
            raise

    def _compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> Tuple[float, List[str]]:
        """Compare two fingerprints and detect modifications."""
        modifications = []

        # Normalize fingerprints
        fp1_norm = cv2.normalize(fp1, None, 0, 1, cv2.NORM_MINMAX)
        fp2_norm = cv2.normalize(fp2, None, 0, 1, cv2.NORM_MINMAX)

        # Calculate NCC
        ncc = cv2.matchTemplate(
            fp1_norm.astype(np.float32),
            fp2_norm.astype(np.float32),
            cv2.TM_CCORR_NORMED
        )[0][0]

        # Calculate SSIM
        ssim = structural_similarity(fp1_norm, fp2_norm, data_range=1.0)

        # Frequency domain comparison
        f1 = np.fft.fft2(fp1_norm)
        f2 = np.fft.fft2(fp2_norm)
        freq_diff = np.abs(f1 - f2)

        # Detect modifications
        if np.mean(freq_diff) > 0.5:
            modifications.append("JPEG compression")
        if np.std(freq_diff) > 0.3:
            modifications.append("Geometric transformation")
        if np.abs(np.mean(fp1) - np.mean(fp2)) > 0.2:
            modifications.append("Intensity modification")

        # Calculate final similarity score
        similarity = 0.6 * ncc + 0.4 * ssim

        return similarity, modifications

    def _detect_modifications(self, fp1: np.ndarray, fp2: np.ndarray) -> List[str]:
        """Detect image modifications."""
        modifications = []
        try:
            # Calculate frequency domain difference
            f1 = np.fft.fft2(fp1)
            f2 = np.fft.fft2(fp2)
            freq_diff = np.abs(f1 - f2)

            # Detect specific modifications
            if np.mean(freq_diff) > 0.5:
                modifications.append("JPEG compression")
            if np.std(freq_diff) > 0.3:
                modifications.append("Geometric transformation")
            if np.abs(np.mean(fp1) - np.mean(fp2)) > 0.2:
                modifications.append("Intensity modification")

        except Exception as e:
            self.logger.error(f"Error in _detect_modifications: {str(e)}")

        return modifications

    def compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> Tuple[float, List[str]]:
        """Compare fingerprints with improved tolerance."""
        try:
            # Ensure same dimensions
            if fp1.shape != fp2.shape:
                fp1 = cv2.resize(fp1, (fp2.shape[1], fp2.shape[0]))

            # Normalize fingerprints
            fp1_norm = cv2.normalize(fp1, None, 0, 1, cv2.NORM_MINMAX)
            fp2_norm = cv2.normalize(fp2, None, 0, 1, cv2.NORM_MINMAX)

            # Calculate multiple similarity metrics
            ncc = float(cv2.matchTemplate(
                fp1_norm.astype(np.float32),
                fp2_norm.astype(np.float32),
                cv2.TM_CCORR_NORMED
            )[0][0])

            ssim = float(structural_similarity(fp1_norm, fp2_norm, data_range=1.0))

            # Wavelet-based comparison
            wavelet_sim = self._compare_wavelets(fp1_norm, fp2_norm)

            # Weighted combination
            similarity = (0.4 * ncc + 0.4 * ssim + 0.2 * wavelet_sim)

            # Detect modifications
            mods = []
            if ncc < 0.4:
                mods.append("Major transformation")
            if ssim < 0.3:
                mods.append("Visual modification")
            if wavelet_sim < 0.25:
                mods.append("Frequency modification")

            return similarity, mods

        except Exception as e:
            self.logger.error(f"Error in compare_fingerprints: {str(e)}")
            return 0.0, ["Error in comparison"]

    def train_verifier(self, authentic_pairs, fake_pairs, epochs=20, batch_size=32):
        """Train the verifier network."""
        X = []
        y = []

        # Prepare authentic pairs
        for orig, embedded in authentic_pairs:
            diff = np.abs(self.extract_fingerprint(embedded) - self.generate_fingerprint(orig))
            X.append(diff)
            y.append(1)

        # Prepare fake pairs
        for orig, fake in fake_pairs:
            diff = np.abs(self.extract_fingerprint(fake) - self.generate_fingerprint(orig))
            X.append(diff)
            y.append(0)

        # Convert to numpy arrays
        X = np.array(X)[..., np.newaxis]
        y = np.array(y)

        # Train the model
        self.verifier.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

    def embed_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
        """Embed fingerprint with stronger and more stable embedding."""
        try:
            # Convert to float32
            image = image.astype(np.float32)

            # Ensure fingerprint size matches image
            h, w = image.shape[:2]
            fp_resized = cv2.resize(fingerprint, (w, h))

            # Create embedding mask focusing on stable regions
            mask = self._calculate_embedding_mask(image)

            # Stronger embedding
            embed_strength = 1.2  # Increased from original
            fp_prepared = fp_resized * mask * embed_strength

            # Embed in multiple wavelet levels
            embedded = np.zeros_like(image)
            for i in range(3):  # For each color channel
                # Wavelet decomposition
                coeffs = pywt.wavedec2(image[:, :, i], 'db1', level=3)
                fp_coeffs = pywt.wavedec2(fp_prepared, 'db1', level=3)

                # Modify coefficients with different strengths
                modified_coeffs = list(coeffs)
                modified_coeffs[0] = coeffs[0] + fp_coeffs[0] * 2.0  # Strong in low frequency

                for j in range(1, len(coeffs)):
                    # Gradually decrease strength in higher frequencies
                    strength = 1.0 / (j + 1)
                    modified_coeffs[j] = tuple(
                        c + f * strength
                        for c, f in zip(coeffs[j], fp_coeffs[j])
                    )

                # Reconstruct
                embedded[:, :, i] = pywt.waverec2(modified_coeffs, 'db1')

            # Normalize and clip
            embedded = np.clip(embedded, 0, 255).astype(np.uint8)
            return embedded

        except Exception as e:
            self.logger.error(f"Error in embed_fingerprint: {str(e)}")
            raise

    def verify_fingerprint(self, image: np.ndarray, original_fingerprint: np.ndarray) -> Tuple[bool, float, List[str]]:
        """Verify fingerprint with improved thresholds."""
        try:
            # Extract fingerprint
            extracted_fp = self.extract_fingerprint(image)

            # Compare fingerprints
            similarity, modifications = self.compare_fingerprints(extracted_fp, original_fingerprint)

            # Adaptive thresholding based on modification type
            base_threshold = 0.35  # Lower base threshold

            if "Major transformation" in modifications:
                threshold = base_threshold * 0.8
            elif "Visual modification" in modifications:
                threshold = base_threshold * 0.9
            else:
                threshold = base_threshold

            # Determine authenticity with inverted logic for unrelated images
            is_authentic = similarity > threshold

            # Additional check for unrelated images
            if "Frequency modification" in modifications and similarity < 0.2:
                is_authentic = False

            return is_authentic, similarity, modifications

        except Exception as e:
            self.logger.error(f"Error in verify_fingerprint: {str(e)}")
            raise
