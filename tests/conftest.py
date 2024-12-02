import pytest
import numpy as np
import cv2
from pathlib import Path
import logging
import yaml


@pytest.fixture
def test_config():
    """Provide test configuration."""
    config = {
        'algorithm_selection': {
            'type': 'amdf',
            'enable_comparison': False
        },
        'algorithm': {
            'fingerprint_size': [64, 64],
            'input_shape': [256, 256, 3],
            'embed_strength': 1.2,
            'similarity_threshold': 0.35
        },
        'directories': {
            'output': 'test_output',
            'database': 'test_database',
            'download': 'test_downloads'
        },
        'testing': {
            'random_seed': 42,
            'deterministic': True
        },
        'resources': {
            'gpu_enabled': False,
            'num_workers': 1
        },
        'logging': {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    return config


@pytest.fixture
def test_logger():
    """Provide test logger."""
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def test_image():
    """Provide test image."""
    # Create a simple test image
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add some patterns for testing
    cv2.circle(image, (128, 128), 64, (255, 0, 0), -1)
    cv2.rectangle(image, (32, 32), (224, 224), (0, 255, 0), 2)
    return image


@pytest.fixture
def test_fingerprint():
    """Provide test fingerprint."""
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def setup_test_dirs(tmp_path):
    """Set up test directories."""
    dirs = ['output', 'database', 'downloads', 'plots', 'reports',
            'fingerprinted', 'manipulated', 'test']

    for dir_name in dirs:
        dir_path = tmp_path / dir_name
        dir_path.mkdir()
        # Create algorithm subdirectories where needed
        if dir_name in ['fingerprinted', 'manipulated', 'plots', 'reports']:
            (dir_path / 'amdf').mkdir()
            (dir_path / 'cdha').mkdir()

    return tmp_path


@pytest.fixture
def private_key():
    """Provide test private key."""
    return "test-private-key-2024"


@pytest.fixture
def create_test_image_set(test_image, tmp_path):
    """Create a set of test images."""
    image_paths = []
    for i in range(5):
        # Create slightly modified versions of the test image
        modified = test_image.copy()
        cv2.putText(modified, f"Test {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        # Save image
        path = tmp_path / f"test_image_{i}.png"
        cv2.imwrite(str(path), modified)
        image_paths.append(str(path))

    return image_paths