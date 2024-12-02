import logging
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import os
from pathlib import Path

import cv2
import numpy as np
import pywt
import tensorflow as tf

from .core.amdf import AdaptiveMultiDomainFingerprinting
from .core.neural_attention import NeuralAttentionEnhancer
from .core.cdha_system import create_cdha_fingerprinter


class BaseNeuralMark(ABC):
    """Base class for NeuralMark algorithms."""

    def __init__(self, private_key: str, logger: logging.Logger, config: dict):
        self.private_key = private_key
        self.logger = logger
        self.config = config
        self.fingerprint_size = tuple(config['algorithm']['fingerprint_size'])
        self.embed_strength = config['algorithm']['embed_strength']

    @abstractmethod
    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Generate fingerprint."""
        pass

    @abstractmethod
    def verify_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> Tuple[bool, float, List[str]]:
        """Verify fingerprint."""
        pass

    @abstractmethod
    def embed_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
        """Embed fingerprint."""
        pass

    @abstractmethod
    def extract_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Extract fingerprint."""
        pass


class AMDFNeuralMark(BaseNeuralMark):
    """AMDF-based implementation of NeuralMark."""

    def __init__(self, private_key: str, logger: logging.Logger, config: dict):
        super().__init__(private_key, logger, config)

        # Initialize AMDF
        self.amdf = AdaptiveMultiDomainFingerprinting(
            config=config,
            logger=logger
        )

        # Initialize neural attention if enabled
        self.use_attention = config['algorithm'].get('neural_attention', {}).get('enabled', False)
        if self.use_attention:
            self.attention_enhancer = NeuralAttentionEnhancer(config, logger)
            self.logger.info("Neural attention enhancement initialized")

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        self.logger.debug("Generating fingerprint using AMDF...")
        try:
            if self.use_attention:
                fingerprint = self.amdf.generate_fingerprint(image)
                enhanced_fp, _ = self.attention_enhancer.enhance_fingerprint(image, fingerprint)
                enhanced_fp = cv2.resize(enhanced_fp, self.fingerprint_size)
                return enhanced_fp
            else:
                fingerprint = self.amdf.generate_fingerprint(image)
                return cv2.resize(fingerprint, self.fingerprint_size)
        except Exception as e:
            self.logger.error(f"Error generating fingerprint: {str(e)}")
            raise

    def verify_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> Tuple[bool, float, List[str]]:
        try:
            extracted_fp = self.extract_fingerprint(image)
            similarity, modifications = self.amdf.compare_fingerprints(extracted_fp, fingerprint)
            threshold = self.config['algorithm']['similarity_threshold']
            is_authentic = similarity > threshold
            return is_authentic, similarity, modifications
        except Exception as e:
            self.logger.error(f"Error verifying fingerprint: {str(e)}")
            raise

    def embed_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
        return self.amdf.embed_fingerprint(image, fingerprint)

    def extract_fingerprint(self, image: np.ndarray) -> np.ndarray:
        return self.amdf.extract_fingerprint(image)


class CDHANeuralMark(BaseNeuralMark):
    """CDHA-based implementation of NeuralMark."""

    def __init__(self, private_key: str, logger: logging.Logger, config: dict):
        super().__init__(private_key, logger, config)
        self.fingerprinter = create_cdha_fingerprinter(config, logger)

    def generate_fingerprint(self, image: np.ndarray) -> np.ndarray:
        return self.fingerprinter.generate_fingerprint(image)

    def verify_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> Tuple[bool, float, List[str]]:
        results = self.fingerprinter.verify_fingerprint(image, fingerprint)
        similarity = 1.0 - results['vulnerability']['vulnerability_score']
        is_authentic = similarity > self.config['algorithm']['similarity_threshold']
        modifications = self._get_modifications(results['vulnerability'])
        return is_authentic, similarity, modifications

    def embed_fingerprint(self, image: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
        try:
            processed_image = cv2.normalize(image.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
            if fingerprint.shape != self.fingerprint_size:
                fingerprint = cv2.resize(fingerprint, self.fingerprint_size)
            embedded = processed_image + fingerprint * self.embed_strength
            embedded = cv2.normalize(embedded, None, 0, 255, cv2.NORM_MINMAX)
            return embedded.astype(np.uint8)
        except Exception as e:
            self.logger.error(f"Error embedding fingerprint: {str(e)}")
            raise

    def extract_fingerprint(self, image: np.ndarray) -> np.ndarray:
        try:
            processed_image = cv2.normalize(image.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
            new_fingerprint = self.generate_fingerprint(processed_image)
            extracted = processed_image - new_fingerprint
            return cv2.normalize(extracted, None, 0, 1, cv2.NORM_MINMAX)
        except Exception as e:
            self.logger.error(f"Error extracting fingerprint: {str(e)}")
            raise

    def _get_modifications(self, vulnerability: dict) -> List[str]:
        modifications = []
        attack_probs = vulnerability['attack_probabilities']
        attack_types = ['compression', 'geometric', 'noise', 'filter', 'crop']
        threshold = self.config['algorithm'].get('attack_threshold', 0.5)

        for attack_type, prob in zip(attack_types, attack_probs):
            if prob > threshold:
                modifications.append(attack_type)

        return modifications


def create_neural_mark(algorithm_type: str, private_key: str, logger: logging.Logger, config: dict) -> BaseNeuralMark:
    """Factory function to create NeuralMark instance.

    Args:
        algorithm_type: Type of algorithm to use ('amdf' or 'cdha')
        private_key: Private key for fingerprinting
        logger: Logger instance
        config: Configuration dictionary

    Returns:
        BaseNeuralMark: Instance of selected algorithm

    Raises:
        ValueError: If algorithm_type is not recognized
    """
    if algorithm_type.lower() == 'amdf':
        return AMDFNeuralMark(private_key, logger, config)
    elif algorithm_type.lower() == 'cdha':
        return CDHANeuralMark(private_key, logger, config)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
