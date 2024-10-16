import os
import requests
import json
from tqdm import tqdm
import random
from alphapunch.enhanced_algorithm import EnhancedAlphaPunch
import logging
from PIL import Image
import argparse
import numpy as np

import signal
import time

UNSPLASH_ACCESS_KEY = "gRMk05LDbP4KvDPQUk2fzf6R8VbCqlcZ_Kvk5TPbVJ0"


def setup_logger():
    logger = logging.getLogger('EnhancedAlphaPunchTester')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
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


def test_enhanced_alphapunch(alphapunch, image_paths, output_dir, timeout=600):  # 10 minutes timeout
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
            is_authentic, similarity, normalized_hamming_distance = alphapunch.verify_fingerprint(output_path, fingerprint)
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
            print(f"Processing of {filename} timed out after {timeout} seconds")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
        finally:
            signal.alarm(0)

    return results


def generate_report(results, report_path):
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Report generated: {report_path}")

    # Print summary
    if results:
        authentic_count = sum(1 for r in results if r['is_authentic'])
        average_psnr = sum(r['psnr'] for r in results) / len(results)
        average_ssim = sum(r['ssim'] for r in results) / len(results)
        average_similarity = sum(r['similarity'] for r in results) / len(results)
        average_hamming_distance = sum(r['normalized_hamming_distance'] for r in results) / len(results)

        print("\nTest Summary:")
        print(f"Total images tested: {len(results)}")
        print(f"Authentic images: {authentic_count} ({authentic_count / len(results) * 100:.2f}%)")
        print(f"Average PSNR: {average_psnr:.2f} dB")
        print(f"Average SSIM: {average_ssim:.4f}")
        print(f"Average Similarity: {average_similarity:.2%}")
        print(f"Average Normalized Hamming Distance: {average_hamming_distance:.2%}")
    else:
        print("\nNo results to report.")


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced AlphaPunch with random Unsplash images")
    parser.add_argument('--num_images', type=int, default=10, help="Number of images to test")
    parser.add_argument('--download_dir', type=str, default='unsplash_images',
                        help="Directory to store downloaded images")
    parser.add_argument('--output_dir', type=str, default='fingerprinted_images',
                        help="Directory to store fingerprinted images")
    parser.add_argument('--report_path', type=str, default='enhanced_alphapunch_report.json',
                        help="Path to save the JSON report")
    args = parser.parse_args()

    global logger
    logger = setup_logger()

    alphapunch = EnhancedAlphaPunch(private_key="your_secret_key_here", logger=logger)

    logger.info(f"Downloading {args.num_images} random images from Unsplash...")
    image_paths = get_random_images(args.num_images, args.download_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Testing Enhanced AlphaPunch on downloaded images...")
    results = test_enhanced_alphapunch(alphapunch, image_paths, args.output_dir)

    logger.info("Generating report...")
    generate_report(results, args.report_path)


if __name__ == "__main__":
    main()
