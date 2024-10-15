import logging
import os
from datetime import datetime
from algorithm import ImprovedAlphaPunch


def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger('AlphaPunch')
    logger.setLevel(logging.DEBUG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'logs/alphapunch_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    logger = setup_logger()
    logger.info("Initializing ImprovedAlphaPunch...")

    private_key = "your_secret_key_here"
    alphapunch = ImprovedAlphaPunch(private_key=private_key, logger=logger, embed_strength=0.5)

    logger.info("Embedding fingerprint...")
    salt = alphapunch.embed_fingerprint("input_image.jpg", "fingerprinted_image.png")

    logger.info("Verifying fingerprint...")
    is_authentic = alphapunch.verify_fingerprint("fingerprinted_image.png", salt)

    logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")
    logger.info("Process complete.")


if __name__ == "__main__":
    main()