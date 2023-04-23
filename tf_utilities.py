import tensorflow as tf



def tf_init(use_gpu, threads, seed):
    """Initialize tensorflow - Set the GPU to be used, the starting seed and available CPU threads"""
    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) > use_gpu, "The requested GPU was not found"

    tf.config.set_visible_devices((gpus[use_gpu] if use_gpu != -1 else []), "GPU")

    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
