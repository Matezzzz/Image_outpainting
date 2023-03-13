import tensorflow as tf






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

class NetworkBuild:
    inp = lambda shape, *args, **kwargs: Network.Input(tf.keras.Input(shape, *args, *kwargs))

    network = lambda val: Network(val)

    dense = lambda u, act = None: TensorflowOp(tf.keras.layers.Dense(u, activation=act))

    @staticmethod
    def flattenInternal(batch_dims, x):
        s = tf.shape(x)
        return tf.reshape(x, tf.concat([s[:batch_dims], [tf.reduce_prod(s[batch_dims:])]], axis=0))

    flatten = lambda batch_dims = 1:   TensorflowOp(lambda x:NetworkBuild.flattenInternal(batch_dims, x))
    batchnorm = lambda: TensorflowOp(tf.keras.layers.BatchNormalization())

    relu = lambda:      TensorflowOp(lambda x:tf.nn.relu(x))
        

    value = lambda: ValueNetworkOp(lambda x, _: x)
    tuple = lambda x, y: x.tuple(y)
    model = lambda: ValueNetworkOp(lambda x, inputs: tf.keras.Model(inputs, x))
    inout = lambda: ValueNetworkOp(lambda x, inputs: (inputs, x))

class ConvNetworkBuild (NetworkBuild):
    conv2d = lambda filters, kernel=3, stride=1, bias=True, padding="same", act=None: TensorflowOp(tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding=padding, use_bias=bias, activation=act))
    conv2ddown = lambda filters:                            ConvNetworkBuild.conv2d(filters, stride=2)
    maxpool = lambda:                                       TensorflowOp(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))


class ResidualNetworkBuild(ConvNetworkBuild):
    residualMain = lambda u, s=1: NetworkOp(lambda x:ConvNetworkBuild.conv2d(u, bias=False, stride=s)(x) >> NetworkBuild.batchnorm() >> NetworkBuild.relu() >> ConvNetworkBuild.conv2d(u, bias=False) >> NetworkBuild.batchnorm())

    residualdown = lambda u: NetworkOp(lambda x:(ConvNetworkBuild.conv2d(u, "relu")(x) + ResidualNetworkBuild.residualMain(u, 2)(x)) >> NetworkBuild.relu())
    residual     = lambda u: NetworkOp(lambda x:(x + ResidualNetworkBuild.residualMain(u)(x)) >> NetworkBuild.relu())
