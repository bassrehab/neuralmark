import logging
import shutil
from pathlib import Path
from typing import List
import datetime

from neuralmark.utils import load_config


class DirectoryCleaner:
    """Manage test directories and cleanup."""

    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger('DirectoryCleaner')
        self.preserved_extensions = {
            'database': ['.json'],
            'downloads': ['.jpg', '.jpeg', '.png']
        }

        if config_path:
            self.config = load_config(config_path)
        else:
            raise ValueError("Config path is required")

    def cleanup(self, pre_run: bool = True):
        """Clean directories while preserving specified files and runs."""
        if not self.config['cleanup']['enabled']:
            self.logger.info("Cleanup disabled in config")
            return

        if pre_run and not self.config['cleanup']['pre_run']:
            self.logger.info("Pre-run cleanup disabled in config")
            return

        if not pre_run and not self.config['cleanup']['post_run']:
            self.logger.info("Post-run cleanup disabled in config")
            return

        self.logger.info("Starting %s cleanup...", "pre-run" if pre_run else "post-run")

        # Clean base directories (downloads, logs)
        self._clean_base_directories()

        # Clean output directory while preserving recent runs
        self._clean_output_directory()

    def _clean_base_directories(self):
        """Clean non-run-specific directories while preserving specified files."""
        base_dirs = ['downloads', 'logs']
        for dir_name in base_dirs:
            try:
                if dir_name in self.preserved_extensions:
                    self._clean_with_preservation(dir_name)
                else:
                    self._clean_directory(dir_name)
            except Exception as e:
                self.logger.error(f"Error cleaning {dir_name}: {str(e)}")

    def _clean_output_directory(self):
        """Clean output directory while preserving recent runs."""
        output_dir = Path(self.config['directories']['base_output'])
        if not output_dir.exists():
            return

        # Get all run directories sorted by creation time
        run_dirs = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
            key=lambda x: x.stat().st_ctime,
            reverse=True
        )

        # Keep only the specified number of recent runs
        runs_to_preserve = self.config['cleanup'].get('preserve_runs', 5)
        for run_dir in run_dirs[runs_to_preserve:]:
            try:
                shutil.rmtree(run_dir)
                self.logger.info(f"Removed old run directory: {run_dir}")
            except Exception as e:
                self.logger.error(f"Error removing run directory {run_dir}: {str(e)}")

    def _clean_directory(self, directory: str):
        """Clear directory contents while preserving the directory itself."""
        dir_path = Path(directory)
        if dir_path.exists():
            for item in dir_path.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    self.logger.error(f"Error cleaning {item}: {str(e)}")
                    