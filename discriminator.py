import tensorflow as tf
from build_network import NetworkBuild as nb, Tensor

def create_discriminator(image_size, residual_blocks):
    def discriminator(x : Tensor):
        y = x\
            >> nb.conv_2d_down(8, activation="relu")\
            >> nb.residual_downscale_sequence([8, 16, 32, 64, 128], residual_blocks)\
            >> nb.conv_2d(128, activation="relu")\
            >> nb.global_average_pooling_2d()\
            >> nb.dense(2, activation="softmax")
        return y
    model = nb.create_model(nb.inp([image_size, image_size, 3]), discriminator)
    model.compile(
        tf.optimizers.Adam(),
        tf.losses.SparseCategoricalCrossentropy(),
        tf.metrics.SparseCategoricalAccuracy(name="discriminator_accuracy")
    )
    return model
