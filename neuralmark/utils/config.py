from dotenv import load_dotenv
import yaml
import os


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    return {
        'UNSPLASH_ACCESS_KEY': os.getenv('UNSPLASH_ACCESS_KEY'),
        'PRIVATE_KEY': os.getenv('PRIVATE_KEY', 'neuralmark-test-key-2024')
    }


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from yaml file and environment variables."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env_vars = load_environment()
    config['unsplash']['access_key'] = env_vars['UNSPLASH_ACCESS_KEY']
    config['private_key'] = env_vars['PRIVATE_KEY']

    return config
