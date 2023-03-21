import os
os.environ["CUDA_VISIBLE_DEVICES"] = open("gpu_to_use.txt").read().splitlines()[0]

import numpy as np
from build_network import ResidualNetworkBuild as nb, TensorflowOp, NetworkOp, Network
import tensorflow as tf
import tensorflow_probability as tfp

from PIL import Image

import argparse

import log_and_save
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from dataset import ImageLoading
from tokenizer import VQVAEModel




def none_or_str(value):
    if value == 'None':
        return None
    return value


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

parser.add_argument("--img_size", default=128, type=int, help="Input image size")
#parser.add_argument("--activation", default="swish", type=str, help="Activation func for convolutions")
#parser.add_argument("--filters", default=32, type=int, help="Base residual layer filters")
#parser.add_argument("--residual_layer_multipliers", default=[1, 2, 3], type=list, help="How many residual layers to use and how to increase number of filters")
#parser.add_argument("--num_res_blocks", default=2, type=int, help="Number of residual blocks in sequence before a downscale")
#parser.add_argument("--embedding_dim", default=32, type=int, help="Embedding dimension for quantizer")
#parser.add_argument("--codebook_size", default=128, type=int, help="Codebook size for quantizer")
#parser.add_argument("--embedding_loss_weight", default=1.0, type=float, help="Scale the embedding loss")
#parser.add_argument("--commitment_loss_weight", default=0.25, type=float, help="Scale for the commitment loss (penalize changing embeddings)")
#parser.add_argument("--entropy_loss_weight", default=0.05, type=float, help="Scale for the entropy loss")
#parser.add_argument("--entropy_loss_temperature", default=0.01, type=float, help="Entropy loss temperature")
parser.add_argument("--hidden_size", default=256, type=int, help="MaskGIT transformer hidden size")
parser.add_argument("--intermediate_size", default=512, type=int, help="MaskGIT transformer intermediate size")
parser.add_argument("--transformer_heads", default=4, type=int, help="MaskGIT transformer heads")
parser.add_argument("--transformer_layers", default=4, type=int, help="MaskGIT transformer layers (how many times do we repeat the self attention block and MLP)")


parser.add_argument("--dataset_location", default=".", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")
parser.add_argument("--tokenizer_dir", default="tokenizer", type=str, help="The tokenizer to use for encoding and decoding")
parser.add_argument("--load_model_dir", default=None, type=none_or_str, help="The model to load")







class TransformerAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, attention_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.layer = tf.keras.layers.MultiHeadAttention(attention_heads, hidden_size, dropout=0.1)

    def call(self, inputs, input_mask):
        return self.layer(inputs, inputs, attention_mask=tf.where(input_mask, 1.0, 0.0))
        
    def get_config(self):
        return super().get_config() | {
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads
        }


class TransformerMLP(tf.keras.layers.Layer):
    def __init__(self, hidden_size, intermediate_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense1 = tf.keras.layers.Dense(self.intermediate_size, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(self.hidden_size)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs):
        return self.layer_norm(inputs + self.dropout(self.dense2(self.dense1(inputs))))

    def get_config(self):
        return super().get_config() | {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        }


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, intermediate_size, attention_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.attention_heads = attention_heads
        self.attention = TransformerAttention(hidden_size, attention_heads)
        self.mlp = TransformerMLP(hidden_size, intermediate_size)
        
    def call(self, inputs, input_mask):
        return self.mlp(self.attention(inputs, input_mask))
        
    def get_config(self):
        return super().get_config() | {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_heads": self.attention_heads
        }


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, codebook_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = self.add_weight("weights", [codebook_size, embed_dim], tf.float32)
    
    @tf.function
    def call(self, x):
        return tf.gather(self.embeddings, x)
    
    @tf.function
    def decode(self, x):
        return x @ tf.transpose(self.embeddings)


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias


def maskgit_embed(token_embed_layer, hidden_size, position_count, intermediate_size, transformer_heads, transformer_layers):
    def maskgit(x):
        mask_tokens = (x == MASK_TOKEN)
        tranformer_inp = (TensorflowOp(token_embed_layer)(x) + nb.embed(position_count, hidden_size)(nb.range(position_count)))\
        >> nb.layerNorm()\
        >> nb.dropout(0.1)
        out = TensorflowOp(TransformerLayer(hidden_size, intermediate_size, transformer_heads))()
    return NetworkOp(maskgit)


def maskgit_transformer(hidden_size, intermediate_size, transformer_heads, transformer_layers):
    #use input mask here
    def transformer(x):
        for _ in range(transformer_layers):
            x = x >> TransformerLayer(hidden_size, intermediate_size, transformer_heads)
        return x
    return NetworkOp(transformer)


def maskgit_final_layer(token_embed_layer : TokenEmbedding, hidden_size):
    return NetworkOp(lambda x: nb.dense(hidden_size, activation="gelu")(x)\
        >> nb.layerNorm()\
        >> TensorflowOp(lambda y: token_embed_layer.decode(y))\
        >> TensorflowOp(BiasLayer())
    )



MASK_TOKEN = -1
class MaskGIT:
    def __init__(self, codebook_size, input_size, hidden_size, intermediate_size, transformer_heads, transformer_layers):
        self._embed_layer = TokenEmbedding(hidden_size, codebook_size)
        input_size = input_size[0] * input_size[1]
        self._model = nb.inp(input_size, dtype=tf.int32)\
            >> maskgit_embed(self._embed_layer, hidden_size, input_size)\
            >> maskgit_transformer(hidden_size, intermediate_size, transformer_heads, transformer_layers)\
            >> maskgit_final_layer(self._embed_layer, hidden_size)\
            >> nb.model()
    
    def get_model(self):
        return self._model
    
    def decode(self, input_tokens, decode_steps):
        #[batch, positions]
        tokens = tf.reshape(input_tokens, [tf.shape(input_tokens)[0], -1])
        token_probs = tf.ones_like(tokens, tf.float32)
        unknown_counts = tf.reduce_sum(input_tokens==MASK_TOKEN, -1)
        for i in range(decode_steps):
            #[batch, positions, tokens]
            logits = self._model(tokens)
            #[batch, positions, tokens]
            dist = tfp.distributions.Categorical(logits)
            
            unknown_tokens = (tokens == MASK_TOKEN)
            sampled_tokens = tf.where(unknown_tokens, dist.sample(), tokens)
            
            
            #gather([batch, positions, tokens], [batch, positions]) -> [batch, positions]
            selected_probs = tf.gather(dist.probs_parameter(), sampled_tokens, axis=-1, batch_dims=2)
            selected_probs = tf.where(unknown_tokens, selected_probs, np.inf)
            
            mask_ratio = tf.math.cos(np.pi / 2 * (i + 1.0) / decode_steps)
            #[batch]
            mask_len = tf.maximum(1, tf.floor(unknown_counts * mask_ratio))
            
            thresholds = tf.gather(tf.sort(selected_probs, -1), mask_len[:, tf.newaxis], axis=-1, batch_dims=1) # type: ignore
            
            tokens = tf.where(selected_probs > thresholds, sampled_tokens, tokens)
            token_probs = tf.where(selected_probs > thresholds, selected_probs, token_probs)
        return tokens, token_probs

    @staticmethod
    def load(dirname):
        mgit = MaskGIT.__new__(MaskGIT)
        mgit._model = tf.keras.models.load_model(dirname, custom_objects={
            "MaskGIT":MaskGIT, "BiasLayer":BiasLayer, "TokenEmbedding":TokenEmbedding,
            "TransformerLayer":TransformerLayer, "TransformerMLP":TransformerMLP, "TransformerAttention":TransformerAttention
        })
        mgit._embed_layer = mgit._model.get_layer(index=1) #type: ignore
        
    
    
    
def main(args):
    #tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    

    LOG_MASKING = 0.9
    class ModelLog:
        _tokenizer : VQVAEModel
        _model : MaskGIT
        
        def __init__(self, tokenizer, model, dataset):
            self._tokenizer = tokenizer
            self._model = model
            self._dataset = iter(dataset)

        def log(self, batch, logs):
            if batch % 400 == 0:
                data, _ = next(self._dataset)
                encoded = self._tokenizer.encode(data)
                mask = tf.random.uniform(tf.shape(encoded)) < LOG_MASKING
                reconstructed_tokens = self._model.decode(tf.where(mask, MASK_TOKEN, encoded), 12)
                reconstructed = self._tokenizer.decode(reconstructed_tokens)
                
                generated = self._model.decode(MASK_TOKEN * tf.ones_like(encoded), 12)
                wandb.log({
                    "images": [wandb.Image(d) for d in data],
                    "reconstruction":[wandb.Image(d) for d in reconstructed],
                    "generated_images": [wandb.Image(d) for d in generated]
                }, commit=False)
    
    
    
    
    image_load = ImageLoading(args.dataset_location, args.img_size, args.places)

    #image_size = np.array([1600, 1200]) // 4
        
    train_dataset = image_load.create_dataset(args.batch_size)#create_dataset(args.batch_size)
    val_dataset = image_load.create_dataset(10)
    
    tokenizer = VQVAEModel.load(args.tokenizer_dir)
    if args.load_model_dir is None:
        model = MaskGIT(tokenizer.codebook_size, tokenizer.latent_space_size, args.hidden_size, args.intermediate_size, args.transformer_heads, args.transformer_layers)
    else:
        model = MaskGIT.load(args.load_model_dir)
    
    wandb_manager = log_and_save.WandbManager("image_outpainting_maskgit")
    wandb_manager.start(args)
    model_log = ModelLog(tokenizer, model, val_dataset)

    model.get_model().fit(train_dataset, epochs=args.epochs, callbacks=[ #type: ignore
        WandbModelCheckpoint("tokenizer", "reconstruction_loss", save_freq=2500),
        tf.keras.callbacks.LambdaCallback(on_batch_end=model_log.log),
        WandbMetricsLogger(50)]
    )
    
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)