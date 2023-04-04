from collections.abc import Callable
from typing import Any
import tensorflow as tf


class Tensor:
    tensor : tf.Tensor

    def __init__(self, tensor : tf.Tensor | Any):
        self.tensor = tensor

    def __rshift__(self, operation) -> "Tensor":
        return Tensor(operation(self.tensor))

    def run_op(self, other, operation) -> "Tensor":
        return Tensor(operation(self.tensor, other.tensor))

    def __add__(self, other):
        return self.run_op(other, lambda x,y:x+y)

    def tuple(self, other):
        return self.run_op(other, lambda x, y:(x, y))

    def get(self):
        return self.tensor

    def __getitem__(self, *args):
        return Tensor(self.tensor.__getitem__(*args)) # type: ignore




class NetworkBuild:
    @classmethod
    def inp(cls, shape, *args, **kwargs):
        return Tensor(tf.keras.layers.Input(shape, *args, **kwargs)) # type: ignore

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

    @classmethod
    def append(cls, tensor, axis):
        def apply(x):
            return tf.concat([x, tensor.get()], axis)
        return apply

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

    @staticmethod
    def batch_norm():
        return tf.keras.layers.BatchNormalization()

    @staticmethod
    def layer_norm():
        return tf.keras.layers.LayerNormalization()

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

    @staticmethod
    def conv_2d(filters, kernel_size=(3, 3), strides=(1, 1), padding="same", **kwargs):
        return tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, **kwargs)

    @classmethod
    def conv_2d_down(cls, filters, kernel_size=(3, 3), strides=(2, 2), padding="same", **kwargs):
        return cls.conv_2d(filters, kernel_size, strides, padding, **kwargs)

    @staticmethod
    def max_pool(*args, **kwargs):
        return tf.keras.layers.MaxPool2D(*args, pool_size=3, strides=2, **kwargs)

    @staticmethod
    def conv_2d_up(filters, kernel_size=(3, 3), strides=(2, 2), padding="same", **kwargs):
        return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding, **kwargs)

    @staticmethod
    def global_average_pooling_2d(*args, **kwargs):
        return tf.keras.layers.GlobalAveragePooling2D(*args, **kwargs)

    @staticmethod
    def residual_link(main_link_func : Callable):
        def apply(x):
            return x + main_link_func(x)
        return apply

    @staticmethod
    def residual_link_ext(main_link_func : Callable, residual_link_func : Callable):
        def apply(x):
            return main_link_func(x) + residual_link_func(x)
        return apply

    @staticmethod
    def residual_normed(main_link_func : Callable, norm = layer_norm):
        def apply(x):
            return norm()(x + main_link_func(x))
        return apply


    @classmethod
    def residual_block_main_link(cls, filters, conv_func = conv_2d, norm_func: Callable = batch_norm, activation: Callable=relu, strides = 1):
        def main_link(x : tf.Tensor):
            out = Tensor(conv_func(filters, use_bias=False, strides=strides)(x))\
                >> norm_func()\
                >> activation\
                >> cls.conv_2d(filters, use_bias=False)\
                >> norm_func()
            return out.get()
        return main_link


    @classmethod
    def residual_block(cls, filters, conv_func = conv_2d, norm_func : Callable = batch_norm, activation: Callable = relu, strides=1):
        def residual_link(x):
            #main_link = cls.residual_main_link(filters, conv_func, norm_func, activation, strides)(x)
            return (x if filters == x.shape[-1] else conv_func(filters, kernel_size=1)(x)) if strides == 1 else conv_func(filters, strides=strides)(x)
        return cls.residual_link_ext(
            cls.residual_block_main_link(filters, conv_func, norm_func, activation, strides),
            residual_link
        )




    @classmethod
    def residual_sequence(cls, blocks, filters, norm_func : Callable = batch_norm, activation: Callable = relu):
        def sequence(x):
            for _ in range(blocks):
                x = cls.residual_block(filters, cls.conv_2d, norm_func, activation)(x)
            return x
        return sequence


    @classmethod
    def residual_downscale_sequence(cls, filter_counts, num_blocks, norm_func : Callable = batch_norm, activation: Callable=relu, return_intermediate=False):
        def downscale(x):
            intermediate = [x]
            for i, fcount in enumerate(filter_counts):
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(x)
                intermediate.append(x)
                if i != len(filter_counts)-1:
                    x = cls.residual_block(fcount, cls.conv_2d, norm_func, activation, strides=2)(x)
            return x if not return_intermediate else intermediate
        return downscale

    @classmethod
    def residual_upscale_sequence(cls, filter_counts, num_blocks, norm_func : Callable = batch_norm, activation: Callable=relu, return_intermediate=False):
        def upscale(x):
            intermediate = [x]
            for i, fcount in enumerate(reversed(filter_counts)):
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(x)
                intermediate.append(x)
                if i != len(filter_counts)-1:
                    x = cls.residual_block(fcount, cls.conv_2d_up, norm_func, activation, strides=2)(x)
            return x if not return_intermediate else intermediate
        return upscale


    @classmethod
    def u_net(cls, filter_counts : list[int], num_blocks : int, norm_func : Callable = batch_norm, activation: Callable=relu):
        def unet(x):
            down = cls.residual_downscale_sequence(filter_counts, num_blocks, norm_func, activation, return_intermediate=True)(x)
            x = cls.residual_sequence(num_blocks, filter_counts[-1], norm_func, activation)(down[-1])
            for y, fcount in zip(down[-2::-1], filter_counts[-2::-1]):
                x = cls.residual_block(fcount, cls.conv_2d_up, norm_func, activation, strides=2)(x)
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(tf.concat([x, y], -1))
            return x
        return unet


    @staticmethod
    def lstm(units, *args, **kwargs):
        return tf.keras.layers.LSTM(units, *args, **kwargs)

    @staticmethod
    def multi_head_self_attention(hidden_size, attention_heads, dropout):
        def apply(x):
            return tf.keras.layers.MultiHeadAttention(attention_heads, hidden_size, hidden_size, dropout)(x, x)
        return apply

    @classmethod
    def transformer_self_attention(cls, hidden_size, attention_heads):
        def apply(x):
            out = Tensor(x) >> cls.residual_link(cls.multi_head_self_attention(hidden_size, attention_heads, dropout=0.1)) >> cls.layer_norm()
            return out.get()
        return apply

    @classmethod
    def transformer_mlp(cls, hidden_size, intermediate_size):
        def apply(x):
            out = Tensor(x) >> cls.residual_link(lambda x: (Tensor(x) >> cls.dense(intermediate_size) >> cls.dense(hidden_size) >> cls.dropout(0.1)).get()) >> cls.layer_norm()
            return out.get()
        return apply

    @classmethod
    def transformer_layer(cls, hidden_size, intermediate_size, attention_heads):
        def apply(x):
            out = Tensor(x) >> cls.transformer_self_attention(hidden_size, attention_heads) >> cls.transformer_mlp(hidden_size, intermediate_size)
            return out.get()
        return apply
