import os
import logging
from typing import Optional
import tensorflow as tf

# Set environment variables
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def configure_tensorflow(config: Optional[dict] = None, logger: Optional[logging.Logger] = None):
    """Configure TensorFlow based on config settings."""
    try:
        if config is None:
            config = {'resources': {'gpu_enabled': False}}

        # GPU configuration
        if not config['resources'].get('gpu_enabled', False):
            tf.config.set_visible_devices([], 'GPU')
            if logger:
                logger.info("GPU disabled - using CPU only")
        else:
            # Configure GPU memory growth
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
        tf.config.threading.set_inter_op_parallelism_threads(
            config['resources'].get('num_workers', 4)
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            config['resources'].get('num_workers', 4)
        )

        # Mixed precision configuration for CDHA
        if config['algorithm_selection'].get('type') == 'cdha' or \
                config['algorithm_selection'].get('enable_comparison', False):
            if tf.config.list_physical_devices('GPU'):
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                if logger:
                    logger.info("Enabled mixed precision for CDHA on GPU")

        # Memory configuration
        if 'memory_limit' in config['resources']:
            memory_limit = int(config['resources']['memory_limit'] * 1024)  # Convert to MB
            tf.config.set_logical_device_configuration(
                tf.config.list_physical_devices('CPU')[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )

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
