import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(config: dict) -> logging.Logger:
    """
    Set up logging configuration with file and console handlers.

    Args:
        config: Configuration dictionary containing logging settings

    Returns:
        Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path(os.getcwd()) / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f'alphapunch_run_{timestamp}.log'

    # Configure logger
    logger = logging.getLogger('AlphaPunch')
    logger.setLevel(config['logging']['level'])

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(str(log_filename))
    file_handler.setFormatter(logging.Formatter(config['logging']['format']))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config['logging']['format']))
    logger.addHandler(console_handler)

    # Log initial setup
    logger.info(f"Logger initialized. Log file: {log_filename}")
    logger.debug(f"Log level set to: {config['logging']['level']}")

    return logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        module_name: Name of the module requesting the logger

    Returns:
        Logger: Module-specific logger
    """
    return logging.getLogger(f'AlphaPunch.{module_name}')
