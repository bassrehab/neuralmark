import unittest
import numpy as np
import cv2
import os
from pathlib import Path
from alphapunch.utils.config import load_config
from alphapunch.utils.logging import setup_logger


class AlphaPunchTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up common test resources."""
        cls.config = load_config('config.yaml')
        cls.logger = setup_logger(cls.config)

        # Create test directories if they don't exist
        cls.test_dirs = ['output', 'database', 'downloads']
        for dir_name in cls.test_dirs:
            Path(dir_name).mkdir(exist_ok=True)

        # Generate test images
        cls.test_images = cls._generate_test_images()

    @classmethod
    def _generate_test_images(cls):
        """Generate a variety of test images."""
        images = {
            'blank': np.zeros((256, 256, 3), dtype=np.uint8),
            'random': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'gradient': cls._create_gradient(),
            'pattern': cls._create_pattern(),
            'text': cls._create_text_image(),
            'noise': cls._create_noise_image()
        }
        return images

    @staticmethod
    def _create_gradient():
        """Create a gradient test image."""
        x = np.linspace(0, 255, 256)
        y = np.linspace(0, 255, 256)
        xx, yy = np.meshgrid(x, y)
        gradient = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.uint8)
        return gradient

    @staticmethod
    def _create_pattern():
        """Create a pattern test image."""
        pattern = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                pattern[i:i + 16, j:j + 16] = [255, 0, 0]
                pattern[i + 16:i + 32, j + 16:j + 32] = [0, 255, 0]
        return pattern

    @staticmethod
    def _create_text_image():
        """Create an image with text."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'AlphaPunch', (50, 128), font, 1, (0, 0, 0), 2)
        return img

    @staticmethod
    def _create_noise_image():
        """Create image with different types of noise."""
        base = np.ones((256, 256, 3), dtype=np.uint8) * 128
        noise = np.random.normal(0, 25, base.shape).astype(np.uint8)
        return cv2.add(base, noise)

    def tearDown(self):
        """Clean up after each test."""
        # Clean test files but keep directories
        for dir_name in self.test_dirs:
            directory = Path(dir_name)
            if directory.exists():
                for item in directory.iterdir():
                    if item.is_file():
                        item.unlink()
