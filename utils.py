import logging
import os
import random
from datetime import datetime, time
from typing import List, Tuple

import cv2
import numpy as np
import requests
import yaml
from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    return {
        'UNSPLASH_ACCESS_KEY': os.getenv('UNSPLASH_ACCESS_KEY'),
        'PRIVATE_KEY': os.getenv('PRIVATE_KEY', 'alphapunch-test-key-2024')
    }


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from yaml file and environment variables."""
    # Load base configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load environment variables
    env_vars = load_environment()

    # Update configuration with environment variables
    config['unsplash']['access_key'] = env_vars['UNSPLASH_ACCESS_KEY']
    config['private_key'] = env_vars['PRIVATE_KEY']

    return config


def setup_logger(config: dict) -> logging.Logger:
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f'alphapunch_run_{timestamp}.log')

    # Configure logger
    logger = logging.getLogger('AlphaPunch')
    logger.setLevel(config['logging']['level'])

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(config['logging']['format']))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config['logging']['format']))
    logger.addHandler(console_handler)

    return logger


def get_test_images(num_images: int, config: dict, logger: logging.Logger) -> List[str]:
    """
    Get test images either from local directory or download from Unsplash.
    """
    download_dir = config['directories']['download']

    # Create directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        logger.info(f"Created download directory: {download_dir}")

    # Get existing images
    existing_images = [
        os.path.join(download_dir, f)
        for f in os.listdir(download_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if len(existing_images) >= num_images:
        logger.info(f"Using {num_images} existing images from {download_dir}")
        return sorted(existing_images)[:num_images]

    # Need to download more images
    images_needed = num_images - len(existing_images)
    logger.info(f"Downloading {images_needed} new images...")

    # Download from Unsplash
    headers = {"Authorization": f"Client-ID {config['unsplash']['access_key']}"}

    try:
        # Make multiple requests if needed (Unsplash has a limit of 30 per request)
        remaining = images_needed
        while remaining > 0:
            batch_size = min(remaining, 30)  # Unsplash maximum per request
            api_url = f"https://api.unsplash.com/photos/random?count={batch_size}"

            response = requests.get(api_url, headers=headers)

            if response.status_code == 200:
                images = response.json()
                for img_data in images:
                    img_url = img_data['urls']['regular']
                    img_id = img_data['id']
                    img_path = os.path.join(download_dir, f"{img_id}.jpg")

                    # Download image if it doesn't exist
                    if not os.path.exists(img_path):
                        img_response = requests.get(img_url)
                        if img_response.status_code == 200:
                            with open(img_path, 'wb') as f:
                                f.write(img_response.content)
                            existing_images.append(img_path)
                            logger.info(f"Downloaded: {img_id}.jpg")

                remaining -= batch_size
            else:
                logger.error(f"Failed to fetch images: {response.status_code}")
                break

            # Add a small delay to respect rate limits
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error downloading images: {str(e)}")

    # Return available images, even if we couldn't download all requested
    return sorted(existing_images)[:num_images]


class ImageManipulator:
    """Class to handle various image manipulations for testing."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.manipulations = {
            'blur': self.blur_image,
            'compress': self.compress_jpeg,
            'rotate': self.rotate_image,
            'crop': self.crop_image,
            'resize': self.resize_image,
            'noise': self.add_noise
        }

    def blur_image(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to image."""
        kernel_size = self.config['testing']['manipulations']['blur_kernel']
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def compress_jpeg(self, image: np.ndarray) -> np.ndarray:
        """Compress image using JPEG compression."""
        quality = self.config['testing']['manipulations']['jpeg_quality']
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Rotate image by specified angle."""
        angle = self.config['testing']['manipulations']['rotation_angle']
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Crop image by percentage from edges."""
        percent = self.config['testing']['manipulations']['crop_percent']
        height, width = image.shape[:2]
        crop_px = int(min(height, width) * percent / 100)
        return image[crop_px:-crop_px, crop_px:-crop_px]

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image by a factor and back to original size."""
        factor = self.config['testing']['manipulations']['resize_factor']
        height, width = image.shape[:2]
        small = cv2.resize(image, (int(width * factor), int(height * factor)))
        return cv2.resize(small, (width, height))

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        std = self.config['testing']['manipulations']['noise_std']
        noise = np.random.normal(0, std, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_manipulation(self, image: np.ndarray, manipulation_name: str) -> np.ndarray:
        """Apply a specific manipulation to an image."""
        if manipulation_name not in self.manipulations:
            raise ValueError(f"Unknown manipulation: {manipulation_name}")

        self.logger.debug(f"Applying manipulation: {manipulation_name}")
        return self.manipulations[manipulation_name](image)

    def apply_random_manipulations(self, image: np.ndarray,
                                   num_manipulations: int = 2) -> Tuple[np.ndarray, List[str]]:
        """Apply random manipulations to an image."""
        manip_names = list(self.manipulations.keys())
        selected = random.sample(manip_names, min(num_manipulations, len(manip_names)))

        result = image.copy()
        self.logger.debug(f"Applying manipulations: {selected}")

        for manip in selected:
            result = self.manipulations[manip](result)

        return result, selected


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_config('config.yaml')

    # Setup logger
    logger = setup_logger(config)

    # Initialize manipulator
    manipulator = ImageManipulator(config, logger)

    # Get test images
    test_images = get_test_images(10, config, logger)

    # Process some test images
    for img_path in test_images:
        logger.info(f"Processing: {os.path.basename(img_path)}")

        # Read original image
        original = cv2.imread(img_path)
        if original is None:
            logger.error(f"Could not read image: {img_path}")
            continue

        # Apply random manipulations
        manipulated, applied_manips = manipulator.apply_random_manipulations(original)

        # Save results
        output_dir = config['directories']['output']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_manipulated.jpg")
        cv2.imwrite(output_path, manipulated)

        logger.info(f"Applied manipulations: {applied_manips}")
        logger.info(f"Saved to: {output_path}")
