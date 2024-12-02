import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache_dir = Path(config['cache']['directory'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = config['cache'].get('enabled', True)
        self.ttl = config['cache'].get('ttl', 3600)  # Time to live in seconds
        self.max_size = config['cache'].get('max_size', 1024)  # Max size in MB

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if not self.enabled:
            return None

        try:
            cache_file = self._get_cache_path(key)
            if not cache_file.exists():
                return None

            # Check if cache entry has expired
            if self._is_expired(cache_file):
                self._remove_cache_file(cache_file)
                return None

            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        except Exception as e:
            self.logger.warning(f"Cache read error for {key}: {str(e)}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """Set item in cache."""
        if not self.enabled:
            return False

        try:
            # Check cache size before adding
            if self._get_cache_size() > self.max_size:
                self._cleanup_cache()

            cache_file = self._get_cache_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            return True

        except Exception as e:
            self.logger.warning(f"Cache write error for {key}: {str(e)}")
            return False

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        hashed_key = hashlib.md5(str(key).encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.cache"

    def _is_expired(self, cache_file: Path) -> bool:
        """Check if cache entry has expired."""
        if not self.ttl:
            return False

        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime > timedelta(seconds=self.ttl)

    def _get_cache_size(self) -> float:
        """Get current cache size in MB."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.cache'))
        return total_size / (1024 * 1024)

    def _cleanup_cache(self):
        """Remove old cache entries."""
        cache_files = sorted(
            self.cache_dir.glob('*.cache'),
            key=lambda x: x.stat().st_mtime
        )

        # Remove oldest files until under max size
        while self._get_cache_size() > self.max_size and cache_files:
            self._remove_cache_file(cache_files.pop(0))

    def _remove_cache_file(self, cache_file: Path):
        """Safely remove cache file."""
        try:
            cache_file.unlink()
        except Exception as e:
            self.logger.warning(f"Error removing cache file {cache_file}: {str(e)}")