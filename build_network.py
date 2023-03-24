from collections.abc import Callable
import tensorflow as tf


class Tensor:
    def __init__(self, tensor):
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




# class BaseNetworkOp:
#     def __init__(self, func, is_type=lambda n:type(n) is Network):
#         self.f = func
#         self.is_type = is_type

#     def __call__(self, val):
#         if not self.is_type(val): raise TypeError("Invalid __call__ parameter")
#         return self.f(val)


# class TensorflowOp(BaseNetworkOp):
#     def __init__(self, func):
#         super().__init__(lambda n: Network(func(n.value), n.inputs))

# class NetworkOp (BaseNetworkOp):
#     def __init__(self, func):
#         super().__init__(lambda n: func(n))

# class ValueNetworkOp (BaseNetworkOp):
#     def __init__(self, func):
#         super().__init__(lambda n: func(n.value, n.inputs))
        

    #def __call__(self, network : Network):
    #    return Network(self.f(network.value), network.inputs)

    #def __rshift__(self, f2):
        #return RecursiveNetworkOp(self.f, f2)



'''
class RecursiveNetworkOp:
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def __call__(self, network : Network):
        return self.f2(self.f1(network))

    def __rshift__(self, f2 : NetworkOp):
        return RecursiveNetworkOp(self.f, f2)
'''


#def tOp(op): return lambda: TensorflowOp(op) 
#def tL(layer, *args, **kwargs): return TensorflowOp(layer(*args, **kwargs))
#def nOp(op): return NetworkOp(op)


class NetworkBuild:
    inp = lambda shape, *args, **kwargs: Tensor(tf.keras.layers.Input(shape, *args, **kwargs))

    @classmethod
    def create_model(cls, input : Tensor, out_func : Callable[[Tensor], Tensor], name=None):
        return tf.keras.Model(input.get(), out_func(input).get(), name=name)
    
    @staticmethod
    def dense(*args, **kwargs): return tf.keras.layers.Dense(*args, **kwargs)
        
    @staticmethod
    def repeat(repeats, axis): return lambda x: tf.repeat(x, repeats, axis)
        
    @staticmethod
    def range(stop): return Tensor(tf.range(stop))    
    
    #dense = lambda u, act = None: TensorflowOp(tf.keras.layers.Dense(u, activation=act))

    # @staticmethod
    # def flattenInternal(batch_dims, x):
    #     s = tf.shape(x)
    #     return tf.reshape(x, tf.concat([s[:batch_dims], [tf.reduce_prod(s[batch_dims:])]], axis=0))

    #flatten = lambda batch_dims = 1:   TensorflowOp(lambda x:NetworkBuild.flattenInternal(batch_dims, x))

    flatten   = lambda: tf.keras.layers.Flatten()


    @staticmethod
    def reshape(new_shape): return lambda x: tf.reshape(x, new_shape)

    
    batchnorm = tf.keras.layers.BatchNormalization
    layerNorm = tf.keras.layers.LayerNormalization
    groupnorm = lambda: tf.keras.layers.GroupNormalization(groups=16) # tfa.layers.GroupNormalization(groups=16)

    @staticmethod
    def dropout(rate): return tf.keras.layers.Dropout(rate)

    @staticmethod
    def embed(input_dim, output_dim): return tf.keras.layers.Embedding(input_dim, output_dim)

    relu = tf.nn.relu
    swish = tf.nn.swish

    #value = lambda: ValueNetworkOp(lambda x, _: x)
    #tuple = lambda x, y: x.tuple(y)
    #model = lambda: ValueNetworkOp(lambda x, inputs: tf.keras.Model(inputs, x))
    #inout = lambda: ValueNetworkOp(lambda x, inputs: (inputs, x))
    @staticmethod
    def model(input : Tensor, output : Tensor): return tf.keras.Model(input.tensor, output.tensor)

nb = NetworkBuild

class ConvNetworkBuild (NetworkBuild):
    @staticmethod
    def conv2D(*args, padding="same", kernel_size=(3, 3), **kwargs):
        return tf.keras.layers.Conv2D(*args, padding=padding, kernel_size=kernel_size, **kwargs)
    @classmethod
    def conv2DDown(cls, *args, **kwargs):
        return cls.conv2D(*args, strides=2, **kwargs)
    @staticmethod
    def maxPool(*args, **kwargs):
        return tf.keras.layers.MaxPool2D(*args, pool_size=3, strides=2, **kwargs)
    @staticmethod
    def conv2DUp(*args, padding="same", kernel_size=(3, 3), strides=2, **kwargs):
        return tf.keras.layers.Conv2DTranspose(*args, padding=padding, kernel_size=kernel_size, strides=strides, **kwargs)
    
    #conv2d = lambda filters, kernel=3, stride=1, bias=True, padding="same", act=None: TensorflowOp(tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding=padding, use_bias=bias, activation=act))
    #conv2ddown = lambda filters:                            ConvNetworkBuild.conv2d(filters, stride=2)
    #maxpool = lambda:                                       TensorflowOp(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

cnb = ConvNetworkBuild

class ResidualNetworkBuild(ConvNetworkBuild):
    @staticmethod
    def residualMainLink(filters, conv_func = cnb.conv2D, norm_func: Callable = nb.batchnorm, activation: Callable=nb.relu, strides = 1):
        def residual(x):
            out = Tensor(conv_func(filters, use_bias=False, strides=strides)(x))\
                >> norm_func()\
                >> activation\
                >> cnb.conv2D(filters, use_bias=False)\
                >> norm_func()
            return out.get()
        return residual
                

    @classmethod
    def residual(cls, filters, conv_func = cnb.conv2D, norm_func : Callable=nb.batchnorm, activation: Callable=nb.relu, strides=1):
        def residual_(x):
            a = cls.residualMainLink(filters, conv_func, norm_func, activation, strides)(x)
            b = (x if filters == x.shape[-1] else conv_func(filters, kernel_size=1)(x)) if strides == 1 else conv_func(filters, strides=strides)(x)
            return a+b
        return residual_

    


    @classmethod
    def residualSequence(cls, blocks, filters, norm_func : Callable =nb.batchnorm, activation: Callable=nb.relu):
        def sequence(x):
            for _ in range(blocks):
                x = cls.residual(filters, cnb.conv2D, norm_func, activation)(x)
            return x
        return sequence


    @classmethod
    def residualDownscaleSequence(cls, filter_counts, num_blocks, norm_func : Callable = nb.batchnorm, activation: Callable=nb.relu):
        def downscale(x):
            for i, fcount in enumerate(filter_counts):
                x = cls.residualSequence(num_blocks, fcount, norm_func, activation)(x)
                if i != len(filter_counts)-1: x = cls.residual(fcount, cnb.conv2D, norm_func, activation, strides=2)(x)
            return x
        return downscale

    @classmethod
    def residualUpscaleSequence(cls, filter_counts, num_blocks, norm_func : Callable = nb.batchnorm, activation: Callable=nb.relu):
        def downscale(x):
            for i, fcount in enumerate(reversed(filter_counts)):
                x = cls.residualSequence(num_blocks, fcount, norm_func, activation)(x)
                if i != len(filter_counts)-1: x = cls.residual(fcount, cnb.conv2DUp, norm_func, activation, strides=2)(x)
            return x
        return downscale