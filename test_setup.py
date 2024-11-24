# test_setup.py
import os
import sys
import tensorflow as tf
from alphapunch.author import ImageAuthor
from utils import load_config, setup_logger


def test_setup():
    print("Testing system setup...")

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print("GPU available:", bool(tf.config.list_physical_devices('GPU')))

    # Test configuration loading
    config = load_config('config.yaml')
    print("Configuration loaded successfully")

    # Test logger setup
    logger = setup_logger(config)
    print("Logger setup successfully")

    # Test author initialization
    author = ImageAuthor(
        private_key=config['private_key'],
        logger=logger,
        config=config
    )
    print("Author initialized successfully")

    print("\nSetup test completed successfully!")


if __name__ == "__main__":
    test_setup()