import logging
import platform
from typing import Tuple, Optional


class GPUManager:
    def __init__(self, logger: logging.Logger, config: dict):
        self.logger = logger
        self.config = config
        self.gpu_available = False
        self.initialized = False

    def setup_gpu(self) -> bool:
        """Setup GPU if available."""
        if self.initialized:
            return self.gpu_available

        try:
            import tensorflow as tf

            # Force CPU if on macOS Metal
            if platform.system() == 'Darwin':
                tf.config.set_visible_devices([], 'GPU')
                self.gpu_available = False
                self.logger.info("Running on CPU due to macOS Metal compatibility")

            else:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Memory growth to avoid taking all GPU memory
                    for gpu in gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                            self.gpu_available = True
                            self.logger.info(f"GPU {gpu.name} initialized with memory growth")
                        except RuntimeError as e:
                            self.logger.warning(f"Error setting up GPU {gpu.name}: {str(e)}")
                else:
                    self.logger.info("No GPU devices available")

        except ImportError:
            self.logger.warning("TensorFlow not available, GPU acceleration disabled")
        except Exception as e:
            self.logger.error(f"Error setting up GPU: {str(e)}")

        self.initialized = True
        return self.gpu_available

    def get_optimal_device(self) -> str:
        """Get the optimal device for computation."""
        if not self.initialized:
            self.setup_gpu()
        return '/GPU:0' if self.gpu_available else '/CPU:0'

    def get_memory_usage(self) -> Optional[Tuple[float, float]]:
        """Get current GPU memory usage (used, total) in MB."""
        if not self.gpu_available:
            return None

        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                import nvidia_smi
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                return (info.used / 1024 ** 2, info.total / 1024 ** 2)
        except:
            pass
        return None