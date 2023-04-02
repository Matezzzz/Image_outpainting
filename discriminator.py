import tensorflow as tf
from build_network import ResidualNetworkBuild as nb, Tensor


class Discriminator:
    def __init__(self, image_size, residual_blocks):
        def discriminator(x : Tensor):
            y = x\
                >> nb.conv2DDown(8, activation="relu")\
                >> nb.residualDownscaleSequence([8, 16, 32, 64, 128], residual_blocks)\
                >> nb.conv2D(128, activation="relu")\
                >> nb.globalAveragePooling2D()\
                >> nb.dense(2, activation="softmax")
            return y
        self._model = nb.create_model(nb.inpT([image_size, image_size, 3]), discriminator)
        self._model.compile(
            tf.optimizers.Adam(),
            tf.losses.SparseCategoricalCrossentropy(),
            tf.metrics.SparseCategoricalAccuracy(name="discriminator_accuracy")
        )
    
    def get_model(self):
        return self._model