import os
GPU_TO_USE = int(open("gpu_to_use.txt").read().splitlines()[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if GPU_TO_USE == -1 else str(GPU_TO_USE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import numpy as np
from build_network import ResidualNetworkBuild as nb, Tensor
import tensorflow as tf
import tensorflow_probability as tfp

import argparse

import log_and_save
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from dataset import ImageLoading
import tokenizer
from tokenizer import VQVAEModel

from utilities import get_tokenizer_fname, get_maskgit_fname


def none_or_str(value):
    if value == 'None':
        return None
    return value


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--hidden_size", default=256, type=int, help="MaskGIT transformer hidden size")
parser.add_argument("--intermediate_size", default=512, type=int, help="MaskGIT transformer intermediate size")
parser.add_argument("--transformer_heads", default=4, type=int, help="MaskGIT transformer heads")
parser.add_argument("--transformer_layers", default=1, type=int, help="MaskGIT transformer layers (how many times do we repeat the self attention block and MLP)")
parser.add_argument("--decode_steps", default=12, type=int, help="MaskGIT decoding steps")
parser.add_argument("--generation_temperature", default=1.0, type=float, help="How random is MaskGIT during decoding")

#parser.add_argument("--train_masking", default=0.9, type=float, help="Percentage of tokens to mask during training")


parser.add_argument("--dataset_location", default=".", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")
parser.add_argument("--load_model", default=False, type=bool, help="Whether to load a maskGIT model")







class TransformerAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, attention_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.layer = tf.keras.layers.MultiHeadAttention(attention_heads, hidden_size, dropout=0.1)
        self.dropout = nb.dropout(0.1)
        self.layer_norm = nb.layerNorm()

    def build(self, input_shape):
        self.layer._build_from_signature(input_shape, input_shape)
        return super().build(input_shape)

    def call(self, inputs): #, input_mask):
        return self.layer_norm(inputs + self.dropout(self.layer(inputs, inputs))) #, attention_mask=input_mask)
        
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
        
    def call(self, inputs):
        return self.mlp(self.attention(inputs))
        
    def get_config(self):
        return super().get_config() | {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_heads": self.attention_heads
        }


# class TokenEmbedding(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, codebook_size, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.embeddings = self.add_weight("weights", [codebook_size, embed_dim], tf.float32)
    
#     def call(self, x) -> tf.Tensor:
#         return tf.gather(self.embeddings, x)
    
#     def decode(self, x):
#         return x @ tf.transpose(self.embeddings)

#!problem - embedding is not forced to be reasonable to predict!

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, codebook_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        #these need to be easily accessed in the training loop
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook = self.add_weight("codebook", [codebook_size, embedding_dim], tf.float32, tf.keras.initializers.TruncatedNormal(0, 0.02))

    def get_embedding(self, tokens):
        #batch dims = [batch, width, height]
        return tf.gather(self.codebook, tokens, axis=0)

    def get_distances(self, values):
        #Dims -> [batch, seq_len, 1, embed] - [codebook_size, embed] ->  [batch, seq_len, codebook_size, embed];
        #reduce_sum -> [batch, seq_len, codebook_size]
        return tf.reduce_sum(tf.square(values[:, :, tf.newaxis, :] - tf.stop_gradient(self.codebook)), -1)

    #values = [batch, seq_len, embed]
    def get_tokens(self, distances):
        return tf.argmin(distances, -1, tf.int32)

    def compute_logits(self, values):
        #[batch, seq_len, embed_dim] @ [codebook_size, embed_dim]
        return values @ tf.transpose(self.codebook) 

    def call(self, inputs):
        mask = inputs==MASK_TOKEN
        return tf.where(mask[:, :, tf.newaxis], 0.0, self.get_embedding(tf.where(mask, 0, inputs)))

    # def call(self, inputs, training, **kwargs):
    #     distances = self.get_distances(inputs)
    #     tokens = self.get_tokens(distances)
    #     embeddings = self.get_embedding(tokens)

    #     #_, _, counts = tf.unique_with_counts(tf.reshape(tokens, [-1]))
    #     #self.add_metric(tf.cast(tf.shape(counts)[0], tf.float32), "codebook_keys_used")

    #     embedding_loss = self.embedding_loss_weight * tf.reduce_mean((tf.stop_gradient(embeddings) - inputs)**2)
    #     commitment_loss = self.commitment_loss_weight * tf.reduce_mean((embeddings - tf.stop_gradient(inputs))**2)
    #     #move all embeddings to their closest vector
    #     entropy_loss = self.entropy_loss_weight * tf.reduce_mean(tf.reduce_min(tf.reshape(distances, [-1, self.codebook_size]), 0)) #self.compute_entropy_loss(distances)

    #     self.add_loss(commitment_loss + embedding_loss+ entropy_loss)
    #     self.add_metric(commitment_loss, "commitment_loss"); self.add_metric(embedding_loss, "embedding_loss"); self.add_metric(entropy_loss, "entropy_loss")
    #     if training: embeddings = inputs + tf.stop_gradient(embeddings - inputs)
    #     return embeddings





class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=input_shape[-1], initializer='zeros', trainable=True)
        
    def call(self, x):
        return x + self.bias


def create_maskgit(codebook_size, hidden_size, position_count, intermediate_size, transformer_heads, transformer_layers):
    def maskgit(tokens):
        embed_layer = TokenEmbedding(codebook_size, hidden_size)
        #def transformer_layer(): return TransformerLayer(hidden_size, intermediate_size, transformer_heads)
        embed = embed_layer(tokens) + nb.embed(position_count, hidden_size)(tf.range(position_count)) # type: ignore
        tranformer_inp = Tensor(embed) >> nb.layerNorm() >> nb.dropout(0.1)
        x = TransformerLayer(hidden_size, intermediate_size, transformer_heads)(tranformer_inp.get())
        for _ in range(transformer_layers-1):
           x = TransformerLayer(hidden_size, intermediate_size, transformer_heads)(x)
            
        out = Tensor(x)\
            >> nb.dense(hidden_size, activation="gelu")\
            >> nb.layerNorm()\
            >> embed_layer.compute_logits\
            >> BiasLayer()
        return out.get()
    return maskgit




MASK_TOKEN = -1
MASKGIT_TRANSFORMER_NAME = "maskgit_transformer"
class MaskGIT(tf.keras.Model):
    def __init__(self, hidden_size, intermediate_size, transformer_heads, transformer_layers, decode_steps, generation_temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = VQVAEModel.load(get_tokenizer_fname(), tokenizer.parser.parse_args([]))
        self._tokenizer.trainable = False
        self._tokenizer.build([None, 128, 128, 3])
        
        
        self.codebook_size = self._tokenizer.codebook_size
        self.input_size_dims = self._tokenizer.latent_space_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.decode_steps = decode_steps
        self.generation_temperature = generation_temperature
        input_size = self.input_size_dims[0] * self.input_size_dims[1]
                
        self._transformer_model = nb.create_model(nb.inpT([input_size], dtype=tf.int32), lambda x: x >> create_maskgit(self.codebook_size, hidden_size, input_size, intermediate_size, transformer_heads, transformer_layers), MASKGIT_TRANSFORMER_NAME)
        self._transformer_model.compile(
            tf.optimizers.Adam(),
            tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            [tf.metrics.SparseCategoricalAccuracy()]
        )
        
        
        self.compile()
        
        msize = (self.input_size_dims[0] // 2, self.input_size_dims[1])
        mask_x = tf.concat([tf.zeros(msize, tf.bool), tf.ones(msize, tf.bool)], 0)
        mask_y = tf.transpose(mask_x)
        self.train_masks = tf.stack([
            mask_x, mask_y, tf.logical_not(mask_x), tf.logical_not(mask_y),
            tf.logical_or(mask_x, mask_y), tf.logical_or(tf.logical_not(mask_x), mask_y),
            tf.logical_or(mask_x, tf.logical_not(mask_y)), tf.logical_or(tf.logical_not(mask_x), tf.logical_not(mask_y))  
        ])
        
        
        
    def reshape_to_seq(self, tokens):
        s = tf.shape(tokens)
        b, w, h = s[0], s[1], s[2]
        return tf.reshape(tokens, [b, w*h])
    
    def reshape_tokens_to_img(self, tokens):
        return tf.reshape(tokens, [tf.shape(tokens)[0], *self.input_size_dims])
    
    def reshape_logits_to_img(self, logits):
        s = tf.shape(logits)
        return tf.reshape(logits, [s[0], *self.input_size_dims, s[2]])
    
    def decode(self, input_tokens, training):
        inp_shape = tf.shape(input_tokens)
        batch_size, w, h, codebook_size = inp_shape[0], inp_shape[1], inp_shape[2], self._transformer_model.output_shape[2]
        token_count = w * h
        #[batch, positions]
        tokens = self.reshape_to_seq(input_tokens)
        token_logits = tf.zeros([batch_size, token_count, codebook_size])
        unknown_counts = tf.reduce_sum(tf.cast(tokens==MASK_TOKEN, tf.int32), -1)
        for i in range(self.decode_steps):
            #[batch, positions, tokens]
            #breakpoint()
            logits = self._transformer_model(tokens, training=training)
            
            unknown_tokens = (tokens == MASK_TOKEN)
            sampled_tokens = tf.reshape(
                tf.random.categorical(tf.reshape(logits, [batch_size * token_count, codebook_size]), num_samples=1, dtype=tf.int32),
                [batch_size, token_count]
            )
            
            
            #gather([batch, positions, tokens], [batch, positions]) -> [batch, positions]
            selected_probs = tf.gather(tf.nn.softmax(logits, axis=-1), sampled_tokens, axis=-1, batch_dims=2)
            selected_probs = tf.where(unknown_tokens, selected_probs, np.inf)
            
            ratio =  (i + 1.0) / self.decode_steps
            mask_ratio = tf.math.cos(np.pi / 2 * ratio)
            #[batch]
            if i == self.decode_steps-1:
                mask_len = tf.zeros_like(unknown_counts)
            else:
                mask_len = tf.maximum(i+1, tf.cast(tf.floor(tf.cast(unknown_counts, tf.float32) * mask_ratio), tf.int32)) # type: ignore
                        
            confidence = tf.math.log(selected_probs) + tfp.distributions.Gumbel(0, 1).sample(tf.shape(selected_probs)) * self.generation_temperature * (1 - ratio)
            
            thresholds = tf.gather(tf.sort(confidence, -1), mask_len[:, tf.newaxis], axis=-1, batch_dims=1) # type: ignore
            
            write_mask = tf.logical_and(confidence >= thresholds, unknown_tokens)
            
            tokens = tf.where(write_mask, sampled_tokens, tokens)
            token_logits = tf.where(write_mask[:, :, tf.newaxis], logits, token_logits) # type: ignore
        return self.reshape_tokens_to_img(tokens), self.reshape_logits_to_img(token_logits)
        #return tf.reshape(tokens, [batch_size, w, h]), tf.reshape(token_logits, [batch_size, w, h, codebook_size])

    def decode_simple(self, input_tokens, training):
        #inp_shape = tf.shape(input_tokens)
        #tokens = tf.reshape(input_tokens, [inp_shape[0], inp_shape[1]*inp_shape[2]])
        logits = self.reshape_logits_to_img(self._transformer_model(self.reshape_to_seq(input_tokens), training=training))
        #logits = tf.reshape(out, [inp_shape[0], inp_shape[1], inp_shape[2], tf.shape(out)[2]])
        return tf.where(input_tokens==MASK_TOKEN, tf.argmax(logits, -1, tf.int32), input_tokens), logits

    def call(self, input_tokens, training):
        return self.decode_simple(input_tokens, training)[1]
        #return self.decode(input_tokens, training)[1] # type: ignore
    
    def train_step(self, data):
        tokens = self._tokenizer.encode(data, training=True)
        mask_idx = tf.random.uniform([tf.shape(tokens)[0]], 0, tf.shape(self.train_masks)[0], tf.int32)
        masks = tf.gather(self.train_masks, mask_idx)
        train_tokens = tf.where(masks, tokens, MASK_TOKEN)
        sample_weight = tf.where(masks, 0.0, 1.0)
        
        with tf.GradientTape() as tape:
            logits = self(train_tokens, training=True)
            loss = self._transformer_model.loss(tokens, logits, sample_weight) # type: ignore
        self._transformer_model.optimizer.minimize(
            loss, self._transformer_model.trainable_variables, tape
        )
        return {'loss':loss} | self._transformer_model.compute_metrics(train_tokens, tokens, logits, sample_weight)
    
    @property
    def downscale_multiplier(self):
        return self._tokenizer.downscale_multiplier
    
    def get_masks(self, data_len):
        mask_idx = tf.range(data_len) % tf.shape(self.train_masks)[0]
        return tf.gather(self.train_masks, mask_idx)    
        
    def to_tokens(self, img, training=False):
        return self._tokenizer.encode(img, training)

    def from_tokens(self, tokens, training=False):
        return self._tokenizer.decode(tokens, training)

    def decode_simple_img(self, tokens, training = False):
        return self.from_tokens(self.decode_simple(tokens, training)[0], training) # type: ignore
    
    def decode_img(self, tokens, training = False):
        return self.from_tokens(self.decode(tokens, training)[0], training) # type: ignore

    @staticmethod
    def load(dirname) -> "MaskGIT":
        return tf.keras.models.load_model(dirname, custom_objects={
            "MaskGIT":MaskGIT, "VQVAEModel":VQVAEModel, "BiasLayer":BiasLayer, "TokenEmbedding":TokenEmbedding,
            "TransformerLayer":TransformerLayer, "TransformerMLP":TransformerMLP, "TransformerAttention":TransformerAttention
        }) #type:ignore
    
    def set_generation_temp(self, temp):
        self.generation_temperature = temp
        
    def get_config(self):
        return super().get_config() | {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "transformer_heads": self.transformer_heads,
            "transformer_layers": self.transformer_layers,
            "decode_steps": self.decode_steps,
            "generation_temperature": self.generation_temperature,
        }
    
    
    
def main(args):
    if GPU_TO_USE == -1: tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    class ModelLog:
        _model : MaskGIT
        
        def __init__(self, model, dataset):
            self._model = model
            self._dataset = iter(dataset)

        def log(self, batch, logs):
            if batch % 200 == 0:
                img = next(self._dataset)
                
                tokens_original = self._model.to_tokens(img)
                
                masks = self._model.get_masks(tf.shape(img)[0])
                tokens_masked = tf.where(masks, tokens_original, MASK_TOKEN)
                
                recreated = self._model.from_tokens(tokens_original)
                
                reconstructed_simple = self._model.decode_simple_img(tokens_masked)
                reconstructed = self._model.decode_img(tokens_masked)
                generated = self._model.decode_img(MASK_TOKEN * tf.ones_like(tokens_original))
                wandb.log({
                    "images": [wandb.Image(d) for d in img],
                    "tokenizer_recreated": [wandb.Image(d) for d in recreated],
                    "reconstruction_simple": [wandb.Image(d) for d in reconstructed_simple],
                    "reconstruction":[wandb.Image(d) for d in reconstructed],
                    "generated_images": [wandb.Image(d) for d in generated]
                }, commit=False)
    




    #tokenizer = VQVAEModel.load(get_tokenizer_fname())
    # def prepare_dataset(img):
    #     #tokens = tf.cast(tokenizer.encode(img), tf.int32)
    #     #mask = tf.random.uniform(tf.shape(tokens)) < tf.random.uniform([tf.shape(tokens)[0], 1, 1])
    #     return tf.where(mask, tf.cast(MASK_TOKEN, tf.int32), tokens), tokens, tf.where(mask, 1.0, 0.0)
    
    image_load = ImageLoading(args.dataset_location, args.img_size, args.places)#, dataset_augmentation_batched=prepare_dataset)
    
    train_dataset = image_load.create_dataset(args.batch_size)
    val_dataset = image_load.create_dataset(8)
    
    
    if not args.load_model:
        model = MaskGIT(args.hidden_size, args.intermediate_size, args.transformer_heads, args.transformer_layers, args.decode_steps, args.generation_temperature)
    else:
        model = MaskGIT.load(get_maskgit_fname())
    
    wandb_manager = log_and_save.WandbManager("image_outpainting_maskgit")
    wandb_manager.start(args)
    model_log = ModelLog(model, val_dataset)

    model.fit(train_dataset, epochs=args.epochs, callbacks=[ #type: ignore
        WandbModelCheckpoint(get_maskgit_fname(), "loss", save_freq=10000),
        tf.keras.callbacks.LambdaCallback(on_batch_end=model_log.log),
        WandbMetricsLogger(25)]
    )
    
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)