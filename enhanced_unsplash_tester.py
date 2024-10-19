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
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logger(config):
    logger = logging.getLogger('EnhancedAlphaPunchTester')
    logger.setLevel(config['logging']['level'])
    handler = logging.StreamHandler()
    formatter = logging.Formatter(config['logging']['format'])
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

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

    while len(downloaded_images) < num_images:
        response = requests.get(
            f"https://api.unsplash.com/photos/random?count={num_images - len(downloaded_images)}",
            headers={"Authorization": f"Client-ID {config['unsplash']['access_key']}"}
        )

        if response.status_code != 200:
            logger.error(f"Failed to fetch images from Unsplash. Status code: {response.status_code}")
            break

        images = response.json()
        for img in images:
            filename = f"{img['id']}.jpg"
            filepath = os.path.join(download_dir, filename)

            if filename in existing_images:
                downloaded_images.append(filepath)
                logger.info(f"Image {filename} already exists. Skipping download.")
            else:
                if download_image(img['urls']['regular'], filepath):
                    downloaded_images.append(filepath)
                    logger.info(f"Downloaded: {filename}")
                else:
                    logger.error(f"Failed to download: {filename}")

    return downloaded_images


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
    print(f"Authentic images: {report['test_summary']['authentic_images']} ({report['test_summary']['authentic_images'] / report['test_summary']['total_images'] * 100:.2f}%)")
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
    args = parser.parse_args()

    global config, logger
    config = load_config(args.config)
    logger = setup_logger(config)

    alphapunch = EnhancedAlphaPunch(
        private_key=config['private_key'],
        logger=logger,
        fingerprint_size=tuple(config['algorithm']['fingerprint_size']),
        embed_strength=config['algorithm']['embed_strength']
    )

    logger.info(f"Downloading {config['testing']['num_images']} random images from Unsplash...")
    image_paths = get_random_images(config['testing']['num_images'], config['directories']['download'])

    if not os.path.exists(config['directories']['output']):
        os.makedirs(config['directories']['output'])

    # Split images for training and testing
    train_images, test_images = train_test_split(image_paths, test_size=0.2, random_state=42)

    logger.info("Creating training data and training verifier...")
    authentic_pairs, fake_pairs = create_training_data(alphapunch, train_images, config['directories']['output'])
    alphapunch.train_verifier(authentic_pairs, fake_pairs)

    logger.info("Testing Enhanced AlphaPunch on test images...")
    results = test_enhanced_alphapunch(alphapunch, test_images, config['directories']['output'])

    robustness_results = []
    if config['testing']['run_robustness']:
        logger.info("Running robustness tests...")
        robustness_results = run_robustness_tests(alphapunch, test_images, config['directories']['output'])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(config['directories']['reports'], f'combined_report_{run_id}.json')
    generate_combined_report(results, robustness_results, report_path)

if __name__ == "__main__":
    main()