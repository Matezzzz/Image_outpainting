from collections.abc import Callable
from typing import Any
import tensorflow as tf


class NBTensor:
    """
    A network build tensor. Holds a variable, on which tensorflow operations can be called as `NBTensor(x) >> tf.exp` (equivalent to `tf.exp(x)`).
    
    Supports chaining multiple operations, can make creating networks more readable and simpler.
    """
    tensor : tf.Tensor

    def __init__(self, tensor : tf.Tensor | Any):
        self.tensor = tensor

    def __rshift__(self, operation) -> "NBTensor":
        """Return NBTensor(operation(self.tensor))"""
        return NBTensor(operation(self.tensor))

    def run_op(self, other, operation) -> "NBTensor":
        """Run an operation on two NBTensors"""
        return NBTensor(operation(self.tensor, other.tensor))

    def __add__(self, other):
        """Add the underlying tensors"""
        return self.run_op(other, lambda x,y:x+y)

    def tuple(self, other):
        """Create a tuple of two tensors"""
        return self.run_op(other, lambda x, y:(x, y))

    def get(self):
        """Get the value wrapped by this class"""
        return self.tensor

    def __getitem__(self, *args):
        """Return the NBTensor after using the [] operator on the child class"""
        return NBTensor(self.tensor.__getitem__(*args))



# pylint: disable=too-many-public-methods
class NetworkBuild:
    @classmethod
    def inp(cls, shape, *args, **kwargs):
        """Create a NBTensor input layer"""
        return NBTensor(tf.keras.layers.Input(shape, *args, **kwargs))

    @staticmethod
    def create_model_io(inp_t : NBTensor | tuple, out_func : Callable[[NBTensor], NBTensor] | Callable):
        """Get tensorflow model input/output using a NBTensor input and a function mapping the input to NBTensor output"""
        inp, out_t = ([i.get() for i in inp_t], out_func(*inp_t)) if isinstance(inp_t, tuple) else (inp_t.get(), out_func(inp_t))
        out = [o.get() for o in out_t] if isinstance(out_t, tuple) else out_t.get()
        return inp, out

    @classmethod
    def create_model(cls, inp : NBTensor | tuple, out_func : Callable[[NBTensor], NBTensor] | Callable, name=None):
        """Create a model with the given input and function mapping input to output"""
        inp, out = cls.create_model_io(inp, out_func)
        return tf.keras.Model(inp, out, name=name)

    @staticmethod
    def model(inp : NBTensor, output : NBTensor):
        """Create a model with given NBTensor input and output"""
        return tf.keras.Model(inp.get(), output.get())

    @staticmethod
    def dense(units : int, activation : str, **kwargs):
        """Create a dense layer with given units and activation"""
        return tf.keras.layers.Dense(units, activation=activation, **kwargs)

    @staticmethod
    def repeat(repeats, axis):
        """Call the tf.repeat function"""
        return lambda x: tf.repeat(x, repeats, axis)

    @staticmethod
    def range(stop):
        """Return a NBTensor of the range specified"""
        return NBTensor(tf.range(stop))

    @staticmethod
    def concat(nb_tensors, axis):
        """Concatenate a tuple of NBTensors along the specified axis"""
        return NBTensor(tf.concat([t.get() for t in nb_tensors], axis))

    @classmethod
    def append(cls, tensor, axis):
        """Return a function that appends the given tensor to it's argument along the provided axis"""
        def apply(x):
            return tf.concat([x, tensor.get()], axis)
        return apply

    @staticmethod
    def flatten():
        """Create a flatten layer"""
        return tf.keras.layers.Flatten()

    @staticmethod
    def reshape(new_shape):
        """Return a function reshaping to the given shape"""
        return lambda x: tf.reshape(x, new_shape)

    @staticmethod
    def batch_norm():
        """Create a batch normalization layer"""
        return tf.keras.layers.BatchNormalization()

    @staticmethod
    def layer_norm():
        """Create a layer normalization layer"""
        return tf.keras.layers.LayerNormalization()

    @staticmethod
    def group_norm(groups=16):
        """Create a group normalization layer"""
        return tf.keras.layers.GroupNormalization(groups)

    @staticmethod
    def dropout(rate):
        """Create a dropout layer"""
        return tf.keras.layers.Dropout(rate)

    @staticmethod
    def embed(input_dim, output_dim):
        """Create an embedding layer"""
        return tf.keras.layers.Embedding(input_dim, output_dim)

    relu = tf.nn.relu

    swish = tf.nn.swish

    @staticmethod
    def conv_2d(filters, kernel_size=(3, 3), strides=(1, 1), padding="same", **kwargs):
        """Create a 2D convolutional layer"""
        return tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, **kwargs)

    @classmethod
    def conv_2d_down(cls, filters, kernel_size=(3, 3), strides=(2, 2), padding="same", **kwargs):
        """Create a 2D convolutional layer, with the default strides being set as 2"""
        return cls.conv_2d(filters, kernel_size, strides, padding, **kwargs)

    @staticmethod
    def max_pool_2d(*args, **kwargs):
        """Create a 2D max pooling layer"""
        return tf.keras.layers.MaxPool2D(*args, pool_size=3, strides=2, **kwargs)

    @staticmethod
    def conv_2d_up(filters, kernel_size=(3, 3), strides=(2, 2), padding="same", **kwargs):
        """Create a 2D transposed convolution layer"""
        return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding, **kwargs)

    @staticmethod
    def global_average_pooling_2d(*args, **kwargs):
        """Create a global average pooling layer"""
        return tf.keras.layers.GlobalAveragePooling2D(*args, **kwargs)

    @staticmethod
    def residual_link(main_link_func : Callable):
        """Create a function `f(x) = x + main_link_func(x)`"""
        return lambda x:x+main_link_func(x)

    @staticmethod
    def residual_link_ext(main_link_func : Callable, residual_link_func : Callable):
        """Create a function `f(x) = main_link_func(x) + residual_link_func(x)"""
        return lambda x: main_link_func(x) + residual_link_func(x)

    @classmethod
    def residual_normed(cls, main_link_func : Callable, norm = layer_norm):
        """Create a function `f(x) = norm()(x + main_link_func(x))`"""
        return lambda x:norm()(cls.residual_link(main_link_func)(x))

    @classmethod
    def residual_block_main_link(cls, filters, conv_func = conv_2d, norm_func: Callable = batch_norm, activation: Callable=relu, strides = 1):
        """A ResNet residual block main link (the part without the skip connection)"""
        def main_link(x : tf.Tensor):
            out = NBTensor(conv_func(filters, use_bias=False, strides=strides)(x))\
                >> norm_func()\
                >> activation\
                >> cls.conv_2d(filters, use_bias=False)\
                >> norm_func()
            return out.get()
        return main_link


    @classmethod
    def residual_block(cls, filters, conv_func = conv_2d, norm_func : Callable = batch_norm, activation: Callable = relu, strides=1):
        """A ResNet residual block. Downscaling convolution is supported if required"""
        def residual_link(x):
            return (x if filters == x.shape[-1] else conv_func(filters, kernel_size=1)(x)) if strides == 1 else conv_func(filters, strides=strides)(x)
        return cls.residual_link_ext(
            cls.residual_block_main_link(filters, conv_func, norm_func, activation, strides),
            residual_link
        )

    @classmethod
    def residual_sequence(cls, blocks, filters, norm_func : Callable = batch_norm, activation: Callable = relu):
        """Multiple ResNet residual blocks in a sequence"""
        def sequence(x):
            for _ in range(blocks):
                x = cls.residual_block(filters, cls.conv_2d, norm_func, activation)(x)
            return x
        return sequence


    @classmethod
    def residual_downscale_sequence(cls, filter_counts, num_blocks, norm_func : Callable = batch_norm, activation: Callable=relu, return_intermediate=False):
        """Multiple ResNet residual blocks in a sequence, then downscale, and repeat"""
        def downscale(x):
            intermediate = [x]
            for i, fcount in enumerate(filter_counts):
                if i != 0:
                    x = cls.residual_block(fcount, cls.conv_2d, norm_func, activation, strides=2)(x)
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(x)
                intermediate.append(x)
            return x if not return_intermediate else intermediate
        return downscale

    @classmethod
    def residual_upscale_sequence(cls, filter_counts, num_blocks, norm_func : Callable = batch_norm, activation: Callable=relu, return_intermediate=False):
        """Multiple ResNet residual blocks in a sequence, then upscale, and repeat"""
        def upscale(x):
            intermediate = [x]
            for i, fcount in enumerate(reversed(filter_counts)):
                if i != 0:
                    x = cls.residual_block(fcount, cls.conv_2d_up, norm_func, activation, strides=2)(x)
                x = cls.residual_sequence(num_blocks, fcount, norm_func, activation)(x)
                intermediate.append(x)
            return x if not return_intermediate else intermediate
        return upscale


    @classmethod
    def u_net(cls, filter_counts : list[int], num_blocks : int, norm_func : Callable = batch_norm, activation: Callable=relu):
        """U-net inspired architecture - first downscale using residual blocks, then upscale, while concatenating the output of residual layers on the same level from downscaling"""
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
        """Create a LSTM layer"""
        return tf.keras.layers.LSTM(units, *args, **kwargs)

    @staticmethod
    def multi_head_self_attention(hidden_size, attention_heads, dropout):
        """Create a multi head self attention layer"""
        return lambda x:tf.keras.layers.MultiHeadAttention(attention_heads, hidden_size, hidden_size, dropout)(x, x)

    @classmethod
    def transformer_self_attention(cls, hidden_size, attention_heads):
        """Create a transformer self attention, with residual link"""
        def apply(x):
            out = NBTensor(x) >> cls.residual_link(cls.multi_head_self_attention(hidden_size, attention_heads, dropout=0.1)) >> cls.layer_norm()
            return out.get()
        return apply

    @classmethod
    def transformer_mlp(cls, hidden_size, intermediate_size):
        """Create a transformer MLP, with residual link"""
        def apply(x):
            out = NBTensor(x) >> cls.residual_link(lambda x: (NBTensor(x) >> cls.dense(intermediate_size, "gelu") >> cls.dense(hidden_size, None) >> cls.dropout(0.1)).get()) >> cls.layer_norm()
            return out.get()
        return apply

    @classmethod
    def transformer_layer(cls, hidden_size, intermediate_size, attention_heads):
        """Apply a transformer self-attention layer, then a MLP"""
        def apply(x):
            out = NBTensor(x) >> cls.transformer_self_attention(hidden_size, attention_heads) >> cls.transformer_mlp(hidden_size, intermediate_size)
            return out.get()
        return apply
