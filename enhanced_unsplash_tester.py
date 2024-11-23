# enhanced_unsplash_tester.py

import os
import requests
import json
import yaml
from tqdm import tqdm
import random
import numpy as np
import cv2
from alphapunch.enhanced_algorithm import EnhancedAlphaPunch
import logging
from PIL import Image
import argparse
import signal
import time
from sklearn.model_selection import train_test_split
from datetime import datetime


def load_config(config_path):
    """Load and validate configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Ensure required sections exist
    required_sections = ['algorithm', 'testing', 'logging', 'directories']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file")

    # Set default values if not specified
    if 'train_test_split' not in config['algorithm']:
        config['algorithm']['train_test_split'] = 0.2
        config['defaults_used'] = ['train_test_split']

    # Validate key parameters
    if config['algorithm']['similarity_threshold'] < 0 or config['algorithm']['similarity_threshold'] > 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    if config['algorithm']['embed_strength'] <= 0:
        raise ValueError("embed_strength must be positive")

    # Add default logging level if not specified
    if 'level' not in config['logging']:
        config['logging']['level'] = 'INFO'

    return config  # Added return statement


def setup_logger(config):
    """
    Set up logging to both file and console with timestamps
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f'alphapunch_run_{timestamp}.log')

    # Create logger
    logger = logging.getLogger('EnhancedAlphaPunchTester')
    logger.setLevel(config['logging']['level'])

    # Remove any existing handlers to avoid duplicate logging
    logger.handlers = []

    # File handler with timestamp
    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(config['logging']['format'])
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initial information
    logger.info(f"Starting new run at {timestamp}")
    logger.info(f"Log file created at: {log_filename}")
    logger.info("Configuration settings:")
    logger.info(f"Testing {config['testing']['num_images']} images")
    logger.info(f"Random seed: {config['testing']['random_seed']}")
    logger.info(f"Embed strength: {config['algorithm']['embed_strength']}")
    logger.info(f"Similarity threshold: {config['algorithm']['similarity_threshold']}")

    # Log any defaults that were used
    if 'defaults_used' in config:
        for param in config['defaults_used']:
            logger.info(f"Using default value for {param}: {config['algorithm'][param]}")

    return logger


def initialize_directories(config):
    """
    Create all required directories if they don't exist.
    Returns a dictionary of created/existing directory paths.
    """
    required_dirs = {
        'download': config['directories']['download'],
        'output': config['directories']['output'],
        'reports': config['directories']['reports'],
        'logs': 'logs'  # Add logs directory to required directories
    }

    logger.info("Initializing directory structure...")

    for dir_name, dir_path in required_dirs.items():
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")
            else:
                logger.debug(f"Directory already exists: {dir_path}")
        except Exception as e:
            logger.error(f"Error creating directory {dir_path}: {str(e)}")
            raise

    # Verify all directories are accessible
    for dir_name, dir_path in required_dirs.items():
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"No write access to {dir_name} directory: {dir_path}")

    return required_dirs


def cleanup_output_directory(output_dir):
    """Clean up fingerprint images and files from previous runs."""
    if not os.path.exists(output_dir):
        return

    logger.info("Cleaning up output directory from previous run...")
    deleted_files = 0
    for filename in os.listdir(output_dir):
        if filename.startswith(('fp_', 'train_', 'robust_')) or filename.endswith('_fingerprint.npy'):
            file_path = os.path.join(output_dir, filename)
            try:
                os.remove(file_path)
                deleted_files += 1
                logger.debug(f"Removed: {filename}")
            except Exception as e:
                logger.error(f"Error removing {filename}: {str(e)}")
    logger.info(f"Cleaned up {deleted_files} files from previous run")


def download_image(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        return True
    return False


def get_random_images(num_images, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    downloaded_images = []
    existing_images = os.listdir(download_dir)

    # First try to use existing images
    for filename in existing_images:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(download_dir, filename)
            downloaded_images.append(filepath)
            if len(downloaded_images) >= num_images:
                logger.info(f"Using {num_images} existing images from {download_dir}")
                return downloaded_images[:num_images]

    # Only download new images if we don't have enough existing ones
    images_needed = num_images - len(downloaded_images)
    if images_needed > 0:
        logger.info(f"Downloading {images_needed} new images...")
        while len(downloaded_images) < num_images:
            response = requests.get(
                f"https://api.unsplash.com/photos/random?count={images_needed}",
                headers={"Authorization": f"Client-ID {config['unsplash']['access_key']}"}
            )

            if response.status_code != 200:
                logger.error(f"Failed to fetch images from Unsplash. Status code: {response.status_code}")
                break

            images = response.json()
            for img in images:
                filename = f"{img['id']}.jpg"
                filepath = os.path.join(download_dir, filename)

                if filename not in existing_images:
                    if download_image(img['urls']['regular'], filepath):
                        downloaded_images.append(filepath)
                        logger.info(f"Downloaded: {filename}")
                    else:
                        logger.error(f"Failed to download: {filename}")

    # Sort the paths to ensure consistent ordering
    downloaded_images.sort()
    return downloaded_images[:num_images]


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def test_enhanced_alphapunch(alphapunch, image_paths, output_dir, timeout=600):
    results = []
    for image_path in tqdm(image_paths, desc="Testing images"):
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"fp_{filename}")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            start_time = time.time()
            fingerprint, psnr, ssim = alphapunch.embed_fingerprint(image_path, output_path)
            embed_time = time.time() - start_time

            start_time = time.time()
            is_authentic, similarity, normalized_hamming_distance = alphapunch.verify_fingerprint(output_path,
                                                                                                  fingerprint)
            verify_time = time.time() - start_time

            results.append({
                'filename': filename,
                'is_authentic': bool(is_authentic),
                'similarity': float(similarity),
                'normalized_hamming_distance': float(normalized_hamming_distance),
                'psnr': float(psnr),
                'ssim': float(ssim),
                'embed_time': embed_time,
                'verify_time': verify_time
            })

        except TimeoutError:
            logger.error(f"Processing of {filename} timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
        finally:
            signal.alarm(0)

    return results


def generate_combined_report(results, robustness_results, report_path):
    report = {
        "test_summary": {
            "total_images": len(results),
            "authentic_images": sum(1 for r in results if r['is_authentic']),
            "average_psnr": sum(r['psnr'] for r in results) / len(results),
            "average_ssim": sum(r['ssim'] for r in results) / len(results),
            "average_similarity": sum(r['similarity'] for r in results) / len(results),
            "average_hamming_distance": sum(r['normalized_hamming_distance'] for r in results) / len(results)
        },
        "detailed_results": results,
        "robustness_results": robustness_results
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Combined report generated: {report_path}")

    print("\nTest Summary:")
    print(f"Total images tested: {report['test_summary']['total_images']}")
    print(
        f"Authentic images: {report['test_summary']['authentic_images']} ({report['test_summary']['authentic_images'] / report['test_summary']['total_images'] * 100:.2f}%)")
    print(f"Average PSNR: {report['test_summary']['average_psnr']:.2f} dB")
    print(f"Average SSIM: {report['test_summary']['average_ssim']:.4f}")
    print(f"Average Similarity: {report['test_summary']['average_similarity']:.2%}")
    print(f"Average Normalized Hamming Distance: {report['test_summary']['average_hamming_distance']:.2%}")


def create_training_data(alphapunch, image_paths, output_dir, num_fake=10):
    authentic_pairs = []
    fake_pairs = []

    for image_path in tqdm(image_paths, desc="Creating training data"):
        img = cv2.imread(image_path)
        fingerprint, _, _ = alphapunch.embed_fingerprint(img, os.path.join(output_dir,
                                                                           f"train_{os.path.basename(image_path)}"))
        authentic_pairs.append((img, alphapunch.amdf.embed_fingerprint(img, fingerprint)))

        for _ in range(num_fake):
            fake_img = img.copy()
            # Apply random manipulations
            if random.choice([True, False]):
                fake_img = cv2.GaussianBlur(fake_img, (5, 5), 0)
            if random.choice([True, False]):
                fake_img = cv2.resize(fake_img, (int(fake_img.shape[1] * 0.9), int(fake_img.shape[0] * 0.9)))
                fake_img = cv2.resize(fake_img, (img.shape[1], img.shape[0]))
            if random.choice([True, False]):
                fake_img = fake_img + np.random.normal(0, 10, fake_img.shape).astype(np.uint8)
            fake_pairs.append((img, fake_img))

    return authentic_pairs, fake_pairs


def create_train_test_split(image_paths, config):
    """Create train/test split ensuring minimum test set size."""
    min_test_images = config['testing'].get('min_test_images', 40)
    total_images = len(image_paths)

    # Get test split ratio with default value if not specified
    test_ratio = config['algorithm'].get('train_test_split', 0.2)

    # Calculate required test ratio to meet minimum test images
    required_test_ratio = min_test_images / total_images

    # Use the larger of configured split ratio and required ratio
    final_test_ratio = max(test_ratio, required_test_ratio)

    # Log the split information
    logger.info(f"Train-test split ratio: {final_test_ratio:.2f}")
    logger.info(f"Min test images required: {min_test_images}")

    # If we can't meet minimum test size, log warning
    if total_images * final_test_ratio < min_test_images:
        logger.warning(
            f"Cannot meet minimum test set size of {min_test_images} images. "
            f"Only have {total_images} total images. Will use {int(total_images * final_test_ratio)} for testing."
        )

    return train_test_split(
        image_paths,
        test_size=final_test_ratio,
        random_state=config['testing'].get('random_seed', 42)
    )


def run_robustness_tests(alphapunch, image_paths, output_dir):
    logger.info("Running robustness tests...")
    robustness_results = []

    manipulations = [
        ("JPEG Compression", lambda img: cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1]),
        ("Gaussian Noise", lambda img: img + np.random.normal(0, 10, img.shape).astype(np.uint8)),
        ("Rotation",
         lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 5, 1),
                                    (img.shape[1], img.shape[0]))),
        ("Scaling", lambda img: cv2.resize(cv2.resize(img, (int(img.shape[1] * 0.9), int(img.shape[0] * 0.9))),
                                           (img.shape[1], img.shape[0]))),
        ("Cropping", lambda img: img[int(img.shape[0] * 0.1):int(img.shape[0] * 0.9),
                                 int(img.shape[1] * 0.1):int(img.shape[1] * 0.9)]),
    ]

    for image_path in tqdm(image_paths, desc="Robustness testing"):
        img = cv2.imread(image_path)
        fingerprint, _, _ = alphapunch.embed_fingerprint(img, os.path.join(output_dir,
                                                                           f"robust_{os.path.basename(image_path)}"))

        for name, manipulation in manipulations:
            manipulated_img = manipulation(img)
            if name == "JPEG Compression":
                manipulated_img = cv2.imdecode(manipulated_img, cv2.IMREAD_COLOR)
            is_authentic, similarity, _ = alphapunch.verify_fingerprint(manipulated_img, fingerprint)

            robustness_results.append({
                'filename': os.path.basename(image_path),
                'manipulation': name,
                'is_authentic': bool(is_authentic),
                'similarity': float(similarity)
            })

    return robustness_results


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced AlphaPunch with random Unsplash images")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration file")
    parser.add_argument('--force-download', action='store_true', help="Force download new images")
    parser.add_argument('--keep-fingerprints', action='store_true', help="Keep fingerprint files from previous runs")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help="Set the logging level")
    args = parser.parse_args()

    # Load config first
    config = load_config(args.config)

    # Override log level if specified in command line
    if args.log_level:
        config['logging']['level'] = args.log_level

    # Setup logger once
    global logger
    logger = setup_logger(config)

    run_start_time = time.time()

    try:
        # Initialize all required directories
        directories = initialize_directories(config)

        # Set random seeds for reproducibility
        np.random.seed(config['testing']['random_seed'])
        random.seed(config['testing']['random_seed'])

        # Clean up previous fingerprint files unless --keep-fingerprints is specified
        if not args.keep_fingerprints:
            cleanup_output_directory(directories['output'])

        # Clear download directory if force-download is specified
        if args.force_download and os.path.exists(directories['download']):
            logger.info("Clearing existing downloaded images...")
            for file in os.listdir(directories['download']):
                os.remove(os.path.join(directories['download'], file))

        alphapunch = EnhancedAlphaPunch(
            private_key=config['private_key'],
            logger=logger,
            config=config
        )

        # Keep trying until we get enough images
        max_attempts = 3
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1}/{max_attempts} to get enough images...")
            image_paths = get_random_images(config['testing']['num_images'], directories['download'])

            if len(image_paths) >= config['testing'].get('min_test_images', 40):
                break
            logger.warning(f"Got only {len(image_paths)} images, retrying...")

        if len(image_paths) < config['testing'].get('min_test_images', 40):
            logger.error(f"Could not get minimum required images after {max_attempts} attempts.")
            return

        logger.info(f"Successfully gathered {len(image_paths)} images.")

        # Log the images being used
        logger.info("Testing with the following images:")
        for path in image_paths:
            logger.info(f"  {os.path.basename(path)}")

        # Use new split function
        train_images, test_images = create_train_test_split(image_paths, config)

        logger.info(f"Split images into {len(train_images)} training and {len(test_images)} testing images")

        logger.info("Creating training data and training verifier...")
        authentic_pairs, fake_pairs = create_training_data(
            alphapunch,
            train_images,
            directories['output']
        )
        alphapunch.train_verifier(authentic_pairs, fake_pairs)

        logger.info("Testing Enhanced AlphaPunch on test images...")
        results = test_enhanced_alphapunch(alphapunch, test_images, directories['output'])

        robustness_results = []
        if config['testing']['run_robustness']:
            logger.info("Running robustness tests...")
            robustness_results = run_robustness_tests(alphapunch, test_images, directories['output'])

        # Generate report
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(directories['reports'], f'combined_report_{run_id}.json')
        generate_combined_report(results, robustness_results, report_path)

        run_end_time = time.time()
        run_duration = run_end_time - run_start_time
        logger.info(f"Run completed in {run_duration:.2f} seconds")

    except Exception as e:
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time
        logger.error(f"Run failed after {run_duration:.2f} seconds")
        logger.error(f"Error: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()
