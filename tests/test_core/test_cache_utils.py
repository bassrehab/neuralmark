import pytest
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from neuralmark.core.cache_utils import CacheManager


class TestCacheManager:
    @pytest.fixture
    def cache_config(self, test_config, tmp_path):
        """Provide cache-specific configuration."""
        test_config['cache'] = {
            'enabled': True,
            'directory': str(tmp_path / 'cache'),
            'ttl': 3600,  # 1 hour
            'max_size': 1024  # 1GB
        }
        return test_config

    @pytest.fixture
    def cache_manager(self, cache_config, test_logger):
        return CacheManager(cache_config, test_logger)

    def test_initialization(self, cache_manager, cache_config):
        """Test cache manager initialization."""
        assert cache_manager.enabled
        assert cache_manager.ttl == cache_config['cache']['ttl']
        assert cache_manager.max_size == cache_config['cache']['max_size']
        assert Path(cache_manager.cache_dir).exists()

    def test_set_and_get(self, cache_manager):
        """Test basic cache set and get operations."""
        test_data = {
            'array': np.random.rand(10, 10),
            'string': 'test string',
            'number': 42
        }

        # Set data
        assert cache_manager.set('test_key', test_data)

        # Get data
        cached_data = cache_manager.get('test_key')
        assert cached_data is not None
        assert isinstance(cached_data, dict)
        assert np.array_equal(cached_data['array'], test_data['array'])
        assert cached_data['string'] == test_data['string']
        assert cached_data['number'] == test_data['number']

    def test_cache_expiration(self, cache_config, test_logger):
        """Test cache entry expiration."""
        # Create cache manager with short TTL
        cache_config['cache']['ttl'] = 1  # 1 second
        manager = CacheManager(cache_config, test_logger)

        # Set data
        manager.set('expire_test', 'test_data')

        # Verify data is cached
        assert manager.get('expire_test') == 'test_data'

        # Wait for expiration
        time.sleep(2)

        # Data should be expired
        assert manager.get('expire_test') is None

    def test_cache_size_limit(self, cache_manager):
        """Test cache size management."""
        # Create large data
        large_data = np.random.rand(1000, 1000)  # About 8MB

        # Set multiple entries
        for i in range(200):  # Should exceed max_size
            cache_manager.set(f'large_data_{i}', large_data)

        # Check cache size
        cache_size = cache_manager._get_cache_size()
        assert cache_size <= cache_manager.max_size

    def test_invalid_cache_access(self, cache_manager):
        """Test handling of invalid cache access."""
        # Try to get non-existent key
        assert cache_manager.get('non_existent_key') is None

        # Try to set None value
        assert not cache_manager.set('none_key', None)

        # Try to set invalid key type
        assert not cache_manager.set(None, 'test_data')

    def test_cache_cleanup(self, cache_manager):
        """Test cache cleanup functionality."""
        # Add some test data
        for i in range(10):
            cache_manager.set(f'test_key_{i}', f'test_data_{i}')

        # Force cleanup
        cache_manager._cleanup_cache()

        # Verify cache size
        assert cache_manager._get_cache_size() <= cache_manager.max_size

    def test_cache_path_generation(self, cache_manager):
        """Test cache file path generation."""
        test_key = 'test_key'
        cache_path = cache_manager._get_cache_path(test_key)

        assert isinstance(cache_path, Path)
        assert cache_path.parent == Path(cache_manager.cache_dir)
        assert cache_path.suffix == '.cache'

    def test_cache_persistence(self, cache_manager):
        """Test cache persistence across instances."""
        # Set data with first instance
        cache_manager.set('persist_test', 'test_data')

        # Create new instance with same config
        new_manager = CacheManager(cache_manager.config, cache_manager.logger)

        # Verify data persists
        assert new_manager.get('persist_test') == 'test_data'

    def test_concurrent_access(self, cache_manager):
        """Test concurrent cache access."""
        import threading

        def cache_operation(key, value):
            cache_manager.set(key, value)
            time.sleep(0.1)
            assert cache_manager.get(key) == value

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=cache_operation,
                args=(f'concurrent_key_{i}', f'value_{i}')
            )
            threads.append(t)

        # Start threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

    @pytest.mark.parametrize("test_data", [
        42,
        "test string",
        [1, 2, 3],
        {'a': 1, 'b': 2},
        np.array([1, 2, 3])
    ])
    def test_different_data_types(self, cache_manager, test_data):
        """Test caching different data types."""
        key = f'type_test_{type(test_data).__name__}'

        # Set data
        assert cache_manager.set(key, test_data)

        # Get data
        cached_data = cache_manager.get(key)

        # Verify data
        if isinstance(test_data, np.ndarray):
            assert np.array_equal(cached_data, test_data)
        else:
            assert cached_data == test_data

    def test_error_handling(self, cache_manager):
        """Test error handling in cache operations."""

        # Test with non-pickleable object
        class NonPickleable:
            def __getstate__(self):
                raise TypeError("Not pickleable")

        # Attempt to cache non-pickleable object
        assert not cache_manager.set('bad_object', NonPickleable())

        # Test with corrupted cache file
        cache_path = cache_manager._get_cache_path('corrupt_test')
        cache_manager.set('corrupt_test', 'test_data')

        # Corrupt the cache file
        with open(cache_path, 'w') as f:
            f.write('corrupted data')

        # Attempt to read corrupted cache
        assert cache_manager.get('corrupt_test') is None
