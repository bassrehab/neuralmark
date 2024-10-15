import logging
import os
from datetime import datetime
from algorithm import RedesignedAlphaPunch


def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create a logger
    logger = logging.getLogger('AlphaPunch')
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'logs/alphapunch_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    logger = setup_logger()
    logger.info("Initializing AlphaPunch...")

    private_key = "your_secret_key_here"
    alphapunch = RedesignedAlphaPunch(private_key=private_key, logger=logger)

    logger.info("Embedding fingerprint...")
    salt = alphapunch.embed_fingerprint("input_image.jpg", "fingerprinted_image.jpg")

    logger.info("Verifying fingerprint...")
    is_authentic = alphapunch.verify_fingerprint("fingerprinted_image.jpg", salt)

    logger.info(f"Verification result: Image is {'authentic' if is_authentic else 'not authentic'}")
    logger.info("Process complete.")


if __name__ == "__main__":
    main()