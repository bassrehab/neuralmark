import hashlib
import logging
import os
import random
from datetime import datetime
import time
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
    """Get test images with improved error handling and retry logic."""
    download_dir = config['directories']['download']
    max_retries = 3
    retry_delay = 2  # seconds

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

    headers = {"Authorization": f"Client-ID {config['unsplash']['access_key']}"}

    try:
        remaining = images_needed
        while remaining > 0:
            batch_size = min(remaining, 30)
            api_url = f"https://api.unsplash.com/photos/random?count={batch_size}"

            # Retry logic for API requests
            for attempt in range(max_retries):
                try:
                    response = requests.get(api_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to fetch images after {max_retries} attempts: {str(e)}")
                        raise
                    time.sleep(retry_delay)

            if response.status_code == 200:
                images = response.json()
                for img_data in images:
                    img_url = img_data['urls']['regular']
                    img_id = img_data['id']
                    img_path = os.path.join(download_dir, f"{img_id}.jpg")

                    if not os.path.exists(img_path):
                        # Retry logic for image downloads
                        for attempt in range(max_retries):
                            try:
                                img_response = requests.get(img_url, timeout=10)
                                img_response.raise_for_status()

                                with open(img_path, 'wb') as f:
                                    f.write(img_response.content)

                                # Verify image can be opened
                                test_img = cv2.imread(img_path)
                                if test_img is None:
                                    raise ValueError("Downloaded image is corrupt")

                                existing_images.append(img_path)
                                logger.info(f"Downloaded: {img_id}.jpg")
                                break
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    logger.error(f"Failed to download image {img_id}: {str(e)}")
                                    continue
                                time.sleep(retry_delay)

                remaining -= batch_size
            else:
                logger.error(f"Failed to fetch images: {response.status_code}")
                break

            # Rate limiting delay
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error downloading images: {str(e)}")

    # Verify minimum images requirement
    if len(existing_images) < num_images:
        logger.warning(f"Could only obtain {len(existing_images)} of {num_images} requested images")

    return sorted(existing_images)[:num_images]


def visualize_attention_maps(image: np.ndarray, attention_maps: List[np.ndarray],
                             save_path: str = None, config: dict = None):
    """Visualize attention maps generated by neural attention module."""
    import matplotlib.pyplot as plt

    num_maps = len(attention_maps)
    fig, axes = plt.subplots(1, num_maps + 1, figsize=(5 * (num_maps + 1), 5))

    # Plot original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot attention maps
    for i, attention_map in enumerate(attention_maps):
        # Normalize attention map
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        norm_map = cv2.normalize(attention_map, None, 0, 1, cv2.NORM_MINMAX)

        # Apply colormap
        colormap = config['visualization']['attention_maps'].get('colormap', 'viridis')
        alpha = config['visualization']['attention_maps'].get('alpha', 0.7)

        axes[i + 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        im = axes[i + 1].imshow(norm_map, cmap=colormap, alpha=alpha)
        axes[i + 1].set_title(f'Attention Layer {i + 1}')
        axes[i + 1].axis('off')
        plt.colorbar(im, ax=axes[i + 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config['visualization']['dpi'])
        plt.close()
    else:
        plt.show()


def _compute_image_hash(image: np.ndarray) -> str:
    """Enhanced image hash computation using both perceptual and cryptographic hashing."""
    # Compute perceptual hash
    resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_low = dct[:8, :8]
    perceptual_hash = (dct_low > np.mean(dct_low)).flatten()

    # Compute cryptographic hash of the perceptual hash
    hasher = hashlib.sha256()
    hasher.update(perceptual_hash.tobytes())

    return hasher.hexdigest()


def apply_random_manipulations(self, image: np.ndarray,
                               num_manipulations: int = 2) -> Tuple[np.ndarray, List[str]]:
    """Apply random manipulations to an image."""
    manip_names = list(self.manipulations.keys())
    selected = random.sample(manip_names, min(num_manipulations, len(manip_names)))

    self.logger.debug(f"Applying manipulations: {selected}")

    result = image.copy()
    for manip in selected:
        result = self.manipulations[manip](result)

    return result, selected


class ImageManipulator:
    """Enhanced image manipulator with configurable parameters."""

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
        """Apply Gaussian blur with configurable parameters."""
        kernel_size = self.config['testing']['manipulations']['blur']['kernel_size']
        sigma = self.config['testing']['manipulations']['blur']['sigma']
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def compress_jpeg(self, image: np.ndarray) -> np.ndarray:
        """Compress image using JPEG with configurable quality."""
        quality = self.config['testing']['manipulations']['compress']['quality']
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """Rotate image by configured angle."""
        angle = self.config['testing']['manipulations']['rotate']['angle']
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Crop image by configured percentage."""
        percent = self.config['testing']['manipulations']['crop']['percent']
        height, width = image.shape[:2]
        crop_px = int(min(height, width) * percent / 100)
        return image[crop_px:-crop_px, crop_px:-crop_px]

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image by configured factor."""
        factor = self.config['testing']['manipulations']['resize']['factor']
        height, width = image.shape[:2]
        small = cv2.resize(image, (int(width * factor), int(height * factor)))
        return cv2.resize(small, (width, height))

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise with configured standard deviation."""
        std = self.config['testing']['manipulations']['noise']['std']
        noise = np.random.normal(0, std, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_manipulation(self, image: np.ndarray, manipulation_name: str) -> np.ndarray:
        """Apply a specific manipulation with error handling."""
        if manipulation_name not in self.manipulations:
            raise ValueError(f"Unknown manipulation: {manipulation_name}")

        try:
            self.logger.debug(f"Applying manipulation: {manipulation_name}")
            return self.manipulations[manipulation_name](image)
        except Exception as e:
            self.logger.error(f"Error applying {manipulation_name}: {str(e)}")
            raise

    def apply_multiple_manipulations(self, image: np.ndarray, manipulation_names: List[str]) -> np.ndarray:
        """Apply multiple manipulations in sequence."""
        result = image.copy()
        for manip_name in manipulation_names:
            result = self.apply_manipulation(result, manip_name)
        return result


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
