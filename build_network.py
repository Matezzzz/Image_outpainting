import tensorflow as tf
import tensorflow_addons as tfa



class Network:
    #add support for late evaluation
    def __init__(self, tensor, inputs = []):
        self.value = tensor
        self.inputs = inputs

    @staticmethod
    def Input(i):
        return Network(i, [i])

    def __rshift__(self, part):
        return part(self)

    def op(self, other, op):
        return Network(op(self.value, other.value), self.mergeInputs(self.inputs, other.inputs))

    def __add__(self, other):
        return self.op(other, lambda x,y:x+y)

    def tuple(self, other):
        return self.op(other, lambda x, y:(x, y))

    @staticmethod
    def mergeInputs(i1, i2):
        return i1 + [i for i in i2 if i not in i1]





class BaseNetworkOp:
    def __init__(self, func, is_type=lambda n:type(n) is Network):
        self.f = func
        self.is_type = is_type

    def __call__(self, val):
        if not self.is_type(val): raise TypeError("Invalid __call__ parameter")
        return self.f(val)


class TensorflowOp(BaseNetworkOp):
    def __init__(self, func):
        super().__init__(lambda n: Network(func(n.value), n.inputs))

class NetworkOp (BaseNetworkOp):
    def __init__(self, func):
        super().__init__(lambda n: func(n))

class ValueNetworkOp (BaseNetworkOp):
    def __init__(self, func):
        super().__init__(lambda n: func(n.value, n.inputs))
        

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


def tOp(op): return lambda: TensorflowOp(op) 
def tL(layer, *args, **kwargs): return TensorflowOp(layer(*args, **kwargs))
def nOp(op): return NetworkOp(op)


class NetworkBuild:
    inp = lambda shape, *args, **kwargs: Network.Input(tf.keras.Input(shape, *args, **kwargs))

    network = lambda val: Network(val)
    
    def dense(*args, **kwargs): return tL(tf.keras.layers.Dense, *args, **kwargs)
        
    @staticmethod
    def repeat(repeats, axis): return tOp(lambda x: tf.repeat(x, repeats, axis))
        
    #dense = lambda u, act = None: TensorflowOp(tf.keras.layers.Dense(u, activation=act))

    # @staticmethod
    # def flattenInternal(batch_dims, x):
    #     s = tf.shape(x)
    #     return tf.reshape(x, tf.concat([s[:batch_dims], [tf.reduce_prod(s[batch_dims:])]], axis=0))

    #flatten = lambda batch_dims = 1:   TensorflowOp(lambda x:NetworkBuild.flattenInternal(batch_dims, x))

    flatten = lambda: tL(tf.keras.layers.Flatten)
    batchnorm = lambda: tL(tf.keras.layers.BatchNormalization)
    groupnorm = lambda: tL(tfa.layers.GroupNormalization, groups=16)


    relu = tOp(tf.nn.relu)
    swish = tOp(tf.nn.swish)

    value = lambda: ValueNetworkOp(lambda x, _: x)
    tuple = lambda x, y: x.tuple(y)
    model = lambda: ValueNetworkOp(lambda x, inputs: tf.keras.Model(inputs, x))
    inout = lambda: ValueNetworkOp(lambda x, inputs: (inputs, x))

nb = NetworkBuild

class ConvNetworkBuild (NetworkBuild):
    @staticmethod
    def conv2D(*args, padding="same", kernel_size=(3, 3), **kwargs):
        return tL(tf.keras.layers.Conv2D, *args, padding=padding, kernel_size=kernel_size, **kwargs)
    @classmethod
    def conv2DDown(cls, *args, **kwargs):
        return cls.conv2D(*args, strides=2, **kwargs)
    @staticmethod
    def maxPool(*args, **kwargs):
        return tL(tf.keras.layers.MaxPool2D, *args, pool_size=3, strides=2, **kwargs)
    @staticmethod
    def conv2DUp(*args, padding="same", kernel_size=(3, 3), strides=2, **kwargs):
        return tL(tf.keras.layers.Conv2DTranspose, *args, padding=padding, kernel_size=kernel_size, strides=strides, **kwargs)
    
    #conv2d = lambda filters, kernel=3, stride=1, bias=True, padding="same", act=None: TensorflowOp(tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding=padding, use_bias=bias, activation=act))
    #conv2ddown = lambda filters:                            ConvNetworkBuild.conv2d(filters, stride=2)
    #maxpool = lambda:                                       TensorflowOp(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

cnb = ConvNetworkBuild

class ResidualNetworkBuild(ConvNetworkBuild):
    @staticmethod
    def residualInternal(filters, conv_func = cnb.conv2D, norm_func = nb.batchnorm, activation=nb.relu, strides = 1):
        return NetworkOp(
            lambda x:
                conv_func(filters, use_bias=False, strides=strides)(x)
                >> norm_func()
                >> activation()
                >> cnb.conv2D(filters, use_bias=False)
                >> norm_func())

    @classmethod
    def residual(cls, filters, conv_func = cnb.conv2D, norm_func=nb.batchnorm, activation=nb.relu, strides=1):
        def residual(x):
            a = cls.residualInternal(filters, conv_func, norm_func, activation, strides)(x)
            b = (x if filters == x.value.shape[-1] else conv_func(filters, kernel_size=1)(x)) if strides == 1 else conv_func(filters, strides=strides)(x)
            return a+b
        return nOp(residual)

    


    @classmethod
    def residualSequence(cls, blocks, filters, norm_func=nb.batchnorm, activation=nb.relu):
        def sequence(x):
            for _ in range(blocks):
                x = cls.residual(filters, cnb.conv2D, norm_func, activation)(x)
            return x
        return nOp(sequence)


    @classmethod
    def residualDownscaleSequence(cls, filter_counts, num_blocks, norm_func = nb.batchnorm, activation=nb.relu):
        def downscale(x):
            for i, fcount in enumerate(filter_counts):
                x = cls.residualSequence(num_blocks, fcount, norm_func, activation)(x)
                if i != len(filter_counts)-1: x = cls.residual(fcount, cnb.conv2D, norm_func, activation, strides=2)(x)
            return x
        return nOp(downscale)
    
    @classmethod
    def residualUpscaleSequence(cls, filter_counts, num_blocks, norm_func = nb.batchnorm, activation=nb.relu):
        def downscale(x):
            for i, fcount in enumerate(reversed(filter_counts)):
                x = cls.residualSequence(num_blocks, fcount, norm_func, activation)(x)
                if i != len(filter_counts)-1: x = cls.residual(fcount, cnb.conv2DUp, norm_func, activation, strides=2)(x)
            return x
        return nOp(downscale)