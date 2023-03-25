import tensorflow as tf
from build_network import ResidualNetworkBuild as nb


class Discriminator:
    def __init__(self, image_size, residual_blocks):
        def discriminator(x):
            return x\
                >> nb.conv2DDown(8, activation="relu")\
                >> nb.residualDownscaleSequence([8, 16, 32, 64, 128], 2)\
                >> nb.conv2D(128, activation="relu")\
                >> nb.flatten()\
                >> nb.dense(128, activation="relu")\
                >> nb.dense(1)
        self._model = nb.create_model(nb.inpT([image_size, image_size, 3]), discriminator)
    
    def get_model(self):
        return self._model