import logging
import shutil
from pathlib import Path

from alphapunch.utils import load_config


class DirectoryCleaner:
    """Manage test directories and cleanup."""

    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger('DirectoryCleaner')
        self.preserved_extensions = {
            'database': ['.json'],
            'downloads': ['.jpg', '.jpeg', '.png']
        }

        if config_path:
            # Load config and update settings if needed
            self.config = load_config(config_path)
            # Could update preserved_extensions or other settings from config

    def cleanup(self, pre_run: bool = True):
        """Clean test directories while preserving specified files."""
        self.logger.info("Starting %s cleanup...", "pre-run" if pre_run else "post-run")

        directories = ['output', 'plots', 'reports', 'fingerprinted', 'manipulated', 'test']

        for dir_name in directories:
            try:
                if dir_name in self.preserved_extensions:
                    self._clean_with_preservation(dir_name)
                else:
                    self._clean_directory(dir_name)
                self.logger.info(f"Cleaned directory: {dir_name}")
            except Exception as e:
                self.logger.error(f"Error cleaning {dir_name}: {str(e)}")

        self.logger.info("Cleanup completed successfully")

    def _clean_directory(self, directory: str):
        """Remove all contents of a directory."""
        dir_path = Path(directory)
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(exist_ok=True)

    def _clean_with_preservation(self, directory: str):
        """Clean directory while preserving specified file types."""
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir()
            return

        preserved_files = []
        preserved_exts = self.preserved_extensions.get(directory, [])

        # Identify files to preserve
        for ext in preserved_exts:
            preserved_files.extend(dir_path.glob(f"*{ext}"))

        # Move preserved files to temporary location
        temp_dir = dir_path.parent / f"temp_{directory}"
        temp_dir.mkdir(exist_ok=True)

        for file in preserved_files:
            shutil.move(str(file), str(temp_dir / file.name))

        # Clean directory
        shutil.rmtree(dir_path)
        dir_path.mkdir()

        # Restore preserved files
        for file in temp_dir.iterdir():
            shutil.move(str(file), str(dir_path / file.name))

        # Remove temporary directory
        shutil.rmtree(temp_dir)


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Clean directories for AlphaPunch')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--pre-run', action='store_true',
                        help='Perform pre-run cleanup')
    parser.add_argument('--post-run', action='store_true',
                        help='Perform post-run cleanup')

    args = parser.parse_args()

    cleaner = DirectoryCleaner(args.config)

    if args.pre_run:
        cleaner.cleanup(pre_run=True)
    elif args.post_run:
        cleaner.cleanup(pre_run=False)
    else:
        cleaner.cleanup(pre_run=True)  # Default to pre-run cleanup
