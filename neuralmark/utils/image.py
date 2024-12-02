import shutil
from pathlib import Path

import cv2
import numpy as np
import requests
import time
import os
import random
from typing import List, Tuple, Optional
import logging


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

    def apply_random_manipulations(self, image: np.ndarray, num_manipulations: int = 2) -> Tuple[np.ndarray, List[str]]:
        """Apply random manipulations to an image."""
        manip_names = list(self.manipulations.keys())
        selected = random.sample(manip_names, min(num_manipulations, len(manip_names)))

        self.logger.debug(f"Applying manipulations: {selected}")

        result = image.copy()
        for manip in selected:
            result = self.manipulations[manip](result)

        return result, selected


def get_test_images(num_images: int, config: dict, logger: logging.Logger) -> List[str]:
    """Get test images with improved error handling and retry logic."""
    download_dir = Path(config['directories']['downloads'])
    test_dir = Path(config['directories']['test'])
    max_retries = 3
    retry_delay = 2

    # Create directories if they don't exist
    download_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Get existing images from both directories
    existing_images = [
        str(f) for f in download_dir.glob('*.[jp][pn][g]')
    ]

    # Copy needed images to test directory
    if len(existing_images) >= num_images:
        selected_images = existing_images[:num_images]
        for img_path in selected_images:
            dest_path = test_dir / Path(img_path).name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
        logger.info(f"Copied {num_images} images to test directory")
        return [str(f) for f in test_dir.glob('*.[jp][pn][g]')][:num_images]

    # Download more images if needed
    images_needed = num_images - len(existing_images)
    logger.info(f"Downloading {images_needed} new images...")

    headers = {"Authorization": f"Client-ID {config['unsplash']['access_key']}"}

    try:
        remaining = images_needed
        while remaining > 0:
            batch_size = min(remaining, 30)
            api_url = f"https://api.unsplash.com/photos/random?count={batch_size}"

            # Initialize response as None before retry loop
            response = None

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

            # Check if we got a valid response
            if response is None or response.status_code != 200:
                logger.error(f"Failed to fetch images: {response.status_code if response else 'No response'}")
                break

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

            # Rate limiting delay
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error downloading images: {str(e)}")

    # Verify minimum images requirement
    if len(existing_images) < num_images:
        logger.warning(f"Could only obtain {len(existing_images)} of {num_images} requested images")

    return sorted(existing_images)[:num_images]


def verify_image(image_path: str) -> Optional[np.ndarray]:
    """Verify image can be opened and is valid."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        if len(img.shape) < 2 or (len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]):
            return None
        return img
    except Exception:
        return None


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to standard format."""
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Convert to RGB if RGBA
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized