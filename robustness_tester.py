import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter, median_filter
from skimage.util import random_noise
import os
import logging
from tqdm import tqdm

from alphapunch.enhanced_algorithm import EnhancedAlphaPunch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RobustnessTester')


def compress_jpeg(image, quality):
    """Compress the image using JPEG compression."""
    is_success, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if is_success:
        return cv2.imdecode(encoded_image, 1)
    else:
        raise Exception("Failed to compress image")


def resize_image(image, scale):
    """Resize the image by a given scale."""
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def crop_image(image, percentage):
    """Crop the image to a given percentage of its original size."""
    height, width = image.shape[:2]
    crop_height = int(height * percentage)
    crop_width = int(width * percentage)
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]


def rotate_image(image, angle):
    """Rotate the image by a given angle."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)


def flip_image(image, direction):
    """Flip the image horizontally or vertically."""
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    elif direction == 'vertical':
        return cv2.flip(image, 0)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")


def adjust_brightness(image, factor):
    """Adjust the brightness of the image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)


def adjust_contrast(image, factor):
    """Adjust the contrast of the image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)


def add_noise(image, noise_type, amount):
    """Add noise to the image."""
    if noise_type == 'gaussian':
        noisy = random_noise(image, mode='gaussian', var=amount ** 2)
    elif noise_type == 'salt_and_pepper':
        noisy = random_noise(image, mode='s&p', amount=amount)
    else:
        raise ValueError("Unsupported noise type")
    return (noisy * 255).astype(np.uint8)


def apply_filter(image, filter_type, params):
    """Apply various filters to the image."""
    if filter_type == 'gaussian_blur':
        return cv2.GaussianBlur(image, (5, 5), params)
    elif filter_type == 'median_blur':
        return cv2.medianBlur(image, params)
    elif filter_type == 'sharpen':
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError("Unsupported filter type")


def test_robustness(alphapunch, original_image, original_fingerprint, manipulations):
    results = []
    for manipulation, params in tqdm(manipulations, desc="Testing manipulations"):
        manipulated_image = manipulation(original_image, *params)
        is_authentic, similarity, _ = alphapunch.verify_fingerprint(manipulated_image, original_fingerprint)
        results.append({
            'manipulation': manipulation.__name__,
            'params': params,
            'is_authentic': is_authentic,
            'similarity': float(similarity)  # Convert to float for JSON serialization
        })
    return results

def run_robustness_tests(alphapunch, image_paths, output_dir):
    robustness_results = {}

    for image_path in image_paths:
        logger.info(f"Running robustness tests on image: {image_path}")
        original_image = cv2.imread(image_path)
        fingerprint, _, _ = alphapunch.embed_fingerprint(original_image, os.path.join(output_dir, f'fingerprinted_{os.path.basename(image_path)}'))

        manipulations = [
            (compress_jpeg, (75,)),
            (compress_jpeg, (50,)),
            (resize_image, (0.75,)),
            (resize_image, (1.25,)),
            (crop_image, (0.9,)),
            (crop_image, (0.75,)),
            (rotate_image, (5,)),
            (rotate_image, (90,)),
            (flip_image, ('horizontal',)),
            (flip_image, ('vertical',)),
            (adjust_brightness, (1.1,)),
            (adjust_brightness, (0.9,)),
            (adjust_contrast, (1.1,)),
            (adjust_contrast, (0.9,)),
            (add_noise, ('gaussian', 0.01)),
            (add_noise, ('salt_and_pepper', 0.01)),
            (apply_filter, ('gaussian_blur', 1)),
            (apply_filter, ('median_blur', 3)),
            (apply_filter, ('sharpen', None)),
        ]

        results = test_robustness(alphapunch, original_image, fingerprint, manipulations)
        robustness_results[image_path] = results

        # Log results
        logger.info(f"Robustness test results for {image_path}:")
        for result in results:
            logger.info(f"Manipulation: {result['manipulation']}, Params: {result['params']}")
            logger.info(f"Authentic: {result['is_authentic']}, Similarity: {result['similarity']:.2%}")
            logger.info("---")

    return robustness_results


def run_robustness_tests_wrapper(alphapunch, image_paths, output_dir):
    return run_robustness_tests(alphapunch, image_paths, output_dir)


def main():
    alphapunch = EnhancedAlphaPunch(private_key="your_secret_key_here")

    # Ensure the output directory exists
    output_dir = 'robustness_test_results'
    os.makedirs(output_dir, exist_ok=True)

    # List of test images
    test_images = ['path_to_image1.jpg', 'path_to_image2.jpg', 'path_to_image3.jpg']

    for image_path in test_images:
        logger.info(f"Testing image: {image_path}")
        original_image = cv2.imread(image_path)
        fingerprint, _, _ = alphapunch.embed_fingerprint(original_image, os.path.join(output_dir,
                                                                                      f'fingerprinted_{os.path.basename(image_path)}'))

        manipulations = [
            (compress_jpeg, (75,)),
            (compress_jpeg, (50,)),
            (resize_image, (0.75,)),
            (resize_image, (1.25,)),
            (crop_image, (0.9,)),
            (crop_image, (0.75,)),
            (rotate_image, (5,)),
            (rotate_image, (90,)),
            (flip_image, ('horizontal',)),
            (flip_image, ('vertical',)),
            (adjust_brightness, (1.1,)),
            (adjust_brightness, (0.9,)),
            (adjust_contrast, (1.1,)),
            (adjust_contrast, (0.9,)),
            (add_noise, ('gaussian', 0.01)),
            (add_noise, ('salt_and_pepper', 0.01)),
            (apply_filter, ('gaussian_blur', 1)),
            (apply_filter, ('median_blur', 3)),
            (apply_filter, ('sharpen', None)),
        ]

        results = test_robustness(alphapunch, original_image, fingerprint, manipulations)

        # Analyze and report results
        logger.info(f"Results for {image_path}:")
        for result in results:
            logger.info(f"Manipulation: {result['manipulation']}, Params: {result['params']}")
            logger.info(f"Authentic: {result['is_authentic']}, Similarity: {result['similarity']:.2%}")
            logger.info("---")

        # Save results to a file
        with open(os.path.join(output_dir, f'results_{os.path.basename(image_path)}.txt'), 'w') as f:
            for result in results:
                f.write(f"Manipulation: {result['manipulation']}, Params: {result['params']}\n")
                f.write(f"Authentic: {result['is_authentic']}, Similarity: {result['similarity']:.2%}\n")
                f.write("---\n")

    logger.info("Robustness testing completed.")


if __name__ == "__main__":
    main()
