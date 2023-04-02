from collections.abc import Callable
from typing import Any
import tensorflow as tf


class Tensor:
    tensor : tf.Tensor
    
    def __init__(self, tensor : tf.Tensor | Any):
        self.tensor = tensor

    def __rshift__(self, op) -> "Tensor":
        return Tensor(op(self.tensor))

    def op(self, other, op) -> "Tensor":
        return Tensor(op(self.tensor, other.tensor))

    def __add__(self, other):
        return self.op(other, lambda x,y:x+y)

    def tuple(self, other):
        return self.op(other, lambda x, y:(x, y))

    @staticmethod
    def mergeInputs(i1, i2):
        return i1 + [i for i in i2 if i not in i1]

    def get(self):
        return self.tensor
    
    def __getitem__(self, *args):
        return Tensor(self.tensor.__getitem__(*args)) # type: ignore




class NetworkBuild:
    @staticmethod
    def inp(shape, *args, **kwargs):
        return tf.keras.layers.Input(shape, *args, **kwargs)
    
    @classmethod
    def inpT(cls, shape, *args, **kwargs):
        return Tensor(cls.inp(shape, *args, **kwargs)) # type: ignore


    @classmethod
    def create_model(cls, inp : Tensor | tuple, out_func : Callable[[Tensor], Tensor] | Callable, name=None):
        if isinstance(inp, tuple):
            return tf.keras.Model([i.get() for i in inp], out_func(*inp).get(), name=name)
        return tf.keras.Model(inp.get(), out_func(inp).get(), name=name)
    
    @staticmethod
    def dense(*args, **kwargs):
        return tf.keras.layers.Dense(*args, **kwargs)
        
    @staticmethod
    def repeat(repeats, axis):
        return lambda x: tf.repeat(x, repeats, axis)
        
    @staticmethod
    def range(stop):
        return Tensor(tf.range(stop))
    
    @staticmethod
    def concat(nb_tensors, axis):
        return Tensor(tf.concat([t.get() for t in nb_tensors], axis))
    
    #dense = lambda u, act = None: TensorflowOp(tf.keras.layers.Dense(u, activation=act))

    # @staticmethod
    # def flattenInternal(batch_dims, x):
    #     s = tf.shape(x)
    #     return tf.reshape(x, tf.concat([s[:batch_dims], [tf.reduce_prod(s[batch_dims:])]], axis=0))

    #flatten = lambda batch_dims = 1:   TensorflowOp(lambda x:NetworkBuild.flattenInternal(batch_dims, x))

    @staticmethod
    def flatten():
        return tf.keras.layers.Flatten()

    @staticmethod
    def reshape(new_shape):
        return lambda x: tf.reshape(x, new_shape)

    
    batchNorm = tf.keras.layers.BatchNormalization
    layerNorm = tf.keras.layers.LayerNormalization
    
    @staticmethod
    def group_norm(groups=16):
        return tf.keras.layers.GroupNormalization(groups)

    @staticmethod
    def dropout(rate):
        return tf.keras.layers.Dropout(rate)

    @staticmethod
    def embed(input_dim, output_dim):
        return tf.keras.layers.Embedding(input_dim, output_dim)

    relu = tf.nn.relu
    swish = tf.nn.swish

    #value = lambda: ValueNetworkOp(lambda x, _: x)
    #tuple = lambda x, y: x.tuple(y)
    #model = lambda: ValueNetworkOp(lambda x, inputs: tf.keras.Model(inputs, x))
    #inout = lambda: ValueNetworkOp(lambda x, inputs: (inputs, x))
    @staticmethod
    def model(inp : Tensor, output : Tensor):
        return tf.keras.Model(inp.get(), output.get())

nb = NetworkBuild

class ConvNetworkBuild (NetworkBuild):
    @staticmethod
    def conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same", **kwargs):
        return tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, **kwargs)
    @classmethod
    def conv2DDown(cls, filters, kernel_size=(3, 3), strides=(2, 2), padding="same", **kwargs):
        return cls.conv2D(filters, kernel_size, strides, padding, **kwargs)
    @staticmethod
    def maxPool(*args, **kwargs):
        return tf.keras.layers.MaxPool2D(*args, pool_size=3, strides=2, **kwargs)
    @staticmethod
    def conv2DUp(filters, kernel_size=(3, 3), strides=(2, 2), padding="same", **kwargs):
        return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding, **kwargs)
    
    @staticmethod
    def globalAveragePooling2D(*args, **kwargs):
        return tf.keras.layers.GlobalAveragePooling2D(*args, **kwargs)
    
    
    #conv2d = lambda filters, kernel=3, stride=1, bias=True, padding="same", act=None: TensorflowOp(tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding=padding, use_bias=bias, activation=act))
    #conv2ddown = lambda filters:                            ConvNetworkBuild.conv2d(filters, stride=2)
    #maxpool = lambda:                                       TensorflowOp(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

cnb = ConvNetworkBuild

class ResidualNetworkBuild(ConvNetworkBuild):
    @staticmethod
    def residual_main_link(filters, conv_func = cnb.conv2D, norm_func: Callable = nb.batchNorm, activation: Callable=nb.relu, strides = 1):
        def residual(x : tf.Tensor):
            out = Tensor(conv_func(filters, use_bias=False, strides=strides)(x))\
                >> norm_func()\
                >> activation\
                >> cnb.conv2D(filters, use_bias=False)\
                >> norm_func()
            return out.get()
        return residual
                

    @classmethod
    def residual(cls, filters, conv_func = cnb.conv2D, norm_func : Callable=nb.batchNorm, activation: Callable=nb.relu, strides=1):
        def residual_(x):
            a = cls.residual_main_link(filters, conv_func, norm_func, activation, strides)(x)
            b = (x if filters == x.shape[-1] else conv_func(filters, kernel_size=1)(x)) if strides == 1 else conv_func(filters, strides=strides)(x)
            return a+b # type: ignore
        return residual_

    


    @classmethod
    def residual_sequence(cls, blocks, filters, norm_func : Callable =nb.batchNorm, activation: Callable=nb.relu):
        def sequence(x):
            for _ in range(blocks):
                x = cls.residual(filters, cnb.conv2D, norm_func, activation)(x)
            return x
        return sequence


    @classmethod
    def residual_downscale_sequence(cls, filter_counts, num_blocks, norm_func : Callable = nb.batchNorm, activation: Callable=nb.relu, return_intermediate=False):
        def downscale(x):
            Ls = [x]
            for i, fcount in enumerate(filter_counts):
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(x)
                Ls.append(x)
                if i != len(filter_counts)-1:
                    x = cls.residual(fcount, cnb.conv2D, norm_func, activation, strides=2)(x)
            return x if not return_intermediate else Ls
        return downscale

    @classmethod
    def residual_upscale_sequence(cls, filter_counts, num_blocks, norm_func : Callable = nb.batchNorm, activation: Callable=nb.relu, return_intermediate=False):
        def upscale(x):
            ls = [x]
            for i, fcount in enumerate(reversed(filter_counts)):
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(x)
                ls.append(x)
                if i != len(filter_counts)-1:
                    x = cls.residual(fcount, cnb.conv2DUp, norm_func, activation, strides=2)(x)
            return x if not return_intermediate else ls
        return upscale
    
    
    @classmethod
    def u_net(cls, filter_counts : list[int], num_blocks : int, norm_func : Callable = nb.batchNorm, activation: Callable=nb.relu):
        def unet(x):
            down = cls.residual_downscale_sequence(filter_counts, num_blocks, norm_func, activation, return_intermediate=True)(x)
            x = cls.residual_sequence(num_blocks, filter_counts[-1], norm_func, activation)(down[-1])
            for y, fcount in zip(down[-2::-1], filter_counts[-2::-1]):
                x = cls.residual(fcount, cls.conv2DUp, norm_func, activation, strides=2)(x)
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(tf.concat([x, y], -1))
            return x
        return unet



class RecurrentNetworkBuild(NetworkBuild):
    @staticmethod
    def lstm(units, *args, **kwargs):
        return tf.keras.layers.LSTM(units, *args, **kwargs)
