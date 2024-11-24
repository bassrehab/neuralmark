import logging
import os
import shutil
from typing import List


class DirectoryCleaner:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize directory cleaner with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()

        # Directories that should be cleaned but preserved
        self.clean_dirs = {
            'output',
            'plots',
            'reports',
            'logs',
            'cache'
        }

        # Directories that should preserve some files
        self.preserve_dirs = {
            'database': ['.json'],  # preserve database files
            'download': ['.jpg', '.jpeg', '.png']  # preserve downloaded images
        }

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from yaml file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('DirectoryCleaner')
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

        return logger

    def cleanup(self, pre_run: bool = True) -> None:
        """Clean directories based on configuration."""
        try:
            self.logger.info(f"Starting {'pre-run' if pre_run else 'post-run'} cleanup...")

            # Clean directories that should be emptied
            for dir_name in self.clean_dirs:
                dir_path = self.config['directories'].get(dir_name)
                if dir_path and os.path.exists(dir_path):
                    self._clean_directory(dir_path)
                    self.logger.info(f"Cleaned directory: {dir_path}")

            # Clean directories with preserved files
            for dir_name, extensions in self.preserve_dirs.items():
                dir_path = self.config['directories'].get(dir_name)
                if dir_path and os.path.exists(dir_path):
                    self._clean_directory_preserve(dir_path, extensions)
                    self.logger.info(f"Cleaned directory (preserving {extensions}): {dir_path}")

            # Special handling for database backups
            if not pre_run:
                self._cleanup_database_backups()

            self.logger.info("Cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

    def _clean_directory(self, directory: str) -> None:
        """Remove all contents of a directory."""
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    self.logger.error(f"Error removing {item_path}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error cleaning directory {directory}: {str(e)}")
            raise

    def _clean_directory_preserve(self, directory: str, preserve_extensions: List[str]) -> None:
        """Clean directory while preserving files with specific extensions."""
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                should_preserve = any(item.lower().endswith(ext) for ext in preserve_extensions)

                if os.path.isfile(item_path) and not should_preserve:
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        except Exception as e:
            self.logger.error(f"Error cleaning directory {directory}: {str(e)}")
            raise

    def _cleanup_database_backups(self) -> None:
        """Cleanup old database backups based on configuration."""
        try:
            db_config = self.config.get('database', {}).get('backup', {})
            if not db_config.get('enabled', False):
                return

            db_dir = self.config['directories']['database']
            keep_last = db_config.get('keep_last', 7)

            # Get all backup files
            backup_files = [f for f in os.listdir(db_dir) if f.startswith('fingerprint_db_backup_')]

            # Sort by date
            backup_files.sort(reverse=True)

            # Remove excess backups
            for backup_file in backup_files[keep_last:]:
                os.remove(os.path.join(db_dir, backup_file))
                self.logger.info(f"Removed old backup: {backup_file}")

        except Exception as e:
            self.logger.error(f"Error cleaning database backups: {str(e)}")
            raise


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