import os
import logging
import platform
from typing import Optional
import tensorflow as tf

# Force CPU on Apple Silicon
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    os.environ['DISABLE_METAL_PLUGIN'] = '1'
    os.environ['DEVICE'] = 'CPU'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def configure_tensorflow(config: Optional[dict] = None, logger: Optional[logging.Logger] = None):
    """Configure TensorFlow based on config settings."""
    try:
        if config is None:
            config = {'resources': {'gpu_enabled': False}}

        # Apple Silicon specific configuration
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            try:
                # Set default memory limit for Apple Silicon
                memory_limit = 1024  # 1GB default
                if 'resources' in config and 'memory_limit' in config['resources']:
                    memory_limit = int(config['resources']['memory_limit'] * 1024)

                # Configure CPU memory
                physical_devices = tf.config.list_physical_devices('CPU')
                if physical_devices:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)

                    # Set virtual device configuration
                    device_config = tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit
                    )
                    tf.config.experimental.set_virtual_device_configuration(
                        physical_devices[0],
                        [device_config]
                    )

                    if logger:
                        logger.info(f"Set memory limit to {memory_limit}MB on Apple Silicon")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not set memory limit on Apple Silicon: {str(e)}")

            # Disable GPU/Metal
            tf.config.set_visible_devices([], 'GPU')

            # Configure threading for Apple Silicon
            num_workers = config['resources'].get('num_workers', 2)
            tf.config.threading.set_inter_op_parallelism_threads(num_workers)
            tf.config.threading.set_intra_op_parallelism_threads(num_workers)

            if logger:
                logger.info("Using CPU on Apple Silicon")
            return

        # Non-Apple Silicon configuration
        # Memory configuration
        if 'resources' in config and 'memory_limit' in config['resources']:
            memory_limit = int(config['resources']['memory_limit'] * 1024)
            try:
                device_config = tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=memory_limit
                )
                tf.config.experimental.set_virtual_device_configuration(
                    tf.config.list_physical_devices('CPU')[0],
                    [device_config]
                )
                if logger:
                    logger.info(f"Set memory limit to {memory_limit}MB")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not set memory limit: {str(e)}")

        # GPU configuration
        if not config['resources'].get('gpu_enabled', False):
            tf.config.set_visible_devices([], 'GPU')
            if logger:
                logger.info("GPU disabled - using CPU only")
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    if logger:
                        logger.info(f"GPU enabled - found {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    if logger:
                        logger.warning(f"Error configuring GPU: {str(e)}")
                    tf.config.set_visible_devices([], 'GPU')
            else:
                if logger:
                    logger.warning("No GPU found - falling back to CPU")

        # Thread configuration
        num_workers = config['resources'].get('num_workers', 4)
        tf.config.threading.set_inter_op_parallelism_threads(num_workers)
        tf.config.threading.set_intra_op_parallelism_threads(num_workers)

        # Mixed precision configuration for CDHA
        if config['algorithm_selection'].get('type') == 'cdha' or \
                config['algorithm_selection'].get('enable_comparison', False):
            if tf.config.list_physical_devices('GPU'):
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                if logger:
                    logger.info("Enabled mixed precision for CDHA on GPU")

        # Deterministic operations for testing
        if config.get('testing', {}).get('deterministic', False):
            tf.config.experimental.enable_op_determinism()
            if logger:
                logger.info("Enabled deterministic operations for testing")

        if logger:
            logger.info("TensorFlow configured successfully")

    except Exception as e:
        if logger:
            logger.error(f"Error configuring TensorFlow: {str(e)}")
        raise


class TensorFlowManager:
    """Manage TensorFlow configurations for different algorithms."""

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger
        self.current_algorithm = None

    def configure_for_algorithm(self, algorithm: str):
        """Configure TensorFlow settings for specific algorithm."""
        try:
            if algorithm == self.current_algorithm:
                return  # Already configured for this algorithm

            if algorithm == 'cdha':
                # CDHA-specific configurations
                if tf.config.list_physical_devices('GPU'):
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                # Set optimized thread settings for CDHA
                tf.config.threading.set_inter_op_parallelism_threads(
                    self.config['resources'].get('num_workers', 4)
                )
            elif algorithm == 'amdf':
                # AMDF-specific configurations
                tf.keras.mixed_precision.set_global_policy('float32')
                # Set thread settings for AMDF
                tf.config.threading.set_inter_op_parallelism_threads(2)

            self.current_algorithm = algorithm
            if self.logger:
                self.logger.info(f"TensorFlow configured for {algorithm.upper()}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error configuring TensorFlow for {algorithm}: {str(e)}")
            raise

    def reset_configuration(self):
        """Reset TensorFlow configuration to default state."""
        try:
            tf.keras.mixed_precision.set_global_policy('float32')
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            self.current_algorithm = None
            if self.logger:
                self.logger.info("TensorFlow configuration reset to default")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error resetting TensorFlow configuration: {str(e)}")
            raise


# Create global TensorFlow manager instance
tf_manager = None


def get_tf_manager(config: Optional[dict] = None, logger: Optional[logging.Logger] = None) -> TensorFlowManager:
    """Get or create TensorFlow manager instance."""
    global tf_manager
    if tf_manager is None and config is not None:
        tf_manager = TensorFlowManager(config, logger)
    return tf_manager


# Run initial configuration
configure_tensorflow()
