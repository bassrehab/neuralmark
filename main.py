import argparse
import logging
from pathlib import Path

from cleanup import DirectoryCleaner
from test_authorship import AuthorshipTester
from neuralmark.utils import load_config, setup_logger


def main():
    parser = argparse.ArgumentParser(description='NeuralMark Image Fingerprinting System')
    # [Previous arguments remain the same]
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Disable cleanup operations')
    parser.add_argument('--preserve-runs', type=int,
                        help='Number of previous runs to preserve')

    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger('NeuralMark')

    try:
        # Load configuration
        config = load_config(args.config)

        # Update cleanup settings from command line
        if args.no_cleanup:
            config['cleanup']['enabled'] = False
        if args.preserve_runs is not None:
            config['cleanup']['preserve_runs'] = args.preserve_runs

        # Create cleaner
        cleaner = DirectoryCleaner(config_path=args.config)

        # Pre-run cleanup if enabled
        if config['cleanup']['enabled'] and config['cleanup']['pre_run']:
            cleaner.cleanup(pre_run=True)

        # Initialize tester
        tester = AuthorshipTester(config_path=args.config)

        # [Rest of the run logic remains the same]

        # Post-run cleanup if enabled
        if config['cleanup']['enabled'] and config['cleanup']['post_run']:
            cleaner.cleanup(pre_run=False)

    except Exception as e:
        logger.error(f"Error running program: {str(e)}")
        raise

    logger.info("Program completed successfully")


if __name__ == "__main__":
    main()
