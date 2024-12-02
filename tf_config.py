import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_FORCE_CPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf


def configure_tensorflow():
    """Configure TensorFlow to use CPU only."""
    try:
        # Disable GPU
        tf.config.set_visible_devices([], 'GPU')

        # Configure memory growth
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        print("TensorFlow configured to use CPU only")
    except RuntimeError as e:
        print(f"Warning: {str(e)}")


# Run configuration immediately on import
configure_tensorflow()