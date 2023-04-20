import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from wandb.keras import WandbModelCheckpoint


from log_and_save import WandbManager, TrainingLog
from dataset import ImageLoading
import tokenizer
from tokenizer import VQVAEModel
from utilities import get_tokenizer_fname, get_maskgit_fname, tf_init
from build_network import NetworkBuild as nb, Tensor


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
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
parser.add_argument("--mask_ratio", default=0.5, type=float, help="Percentage of tokens to mask during training")



parser.add_argument("--dataset_location", default="data", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")
parser.add_argument("--load_model", default=False, type=bool, help="Whether to load a maskGIT model")





class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, codebook_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)

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

    def call(self, inputs, *args, **kwargs):
        mask = inputs==MASK_TOKEN
        return tf.where(mask[:, :, tf.newaxis], 0.0, self.get_embedding(tf.where(mask, 0, inputs)))


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None

    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, inputs, *args, **kwargs):
        return inputs + self.bias


def create_maskgit(codebook_size, hidden_size, position_count, intermediate_size, transformer_heads, transformer_layers):
    def maskgit(tokens):
        embed_layer = TokenEmbedding(codebook_size, hidden_size)
        #def transformer_layer(): return TransformerLayer(hidden_size, intermediate_size, transformer_heads)
        embed = embed_layer(tokens) + nb.embed(position_count, hidden_size)(tf.range(position_count)) # type: ignore
        x = Tensor(embed) >> nb.layer_norm() >> nb.dropout(0.1)
        for _ in range(transformer_layers):
            x = x >> nb.transformer_layer(hidden_size, intermediate_size, transformer_heads)

        out = x\
            >> nb.dense(hidden_size, activation="gelu")\
            >> nb.layer_norm()\
            >> nb.dense(codebook_size, activation=None)
            #>> embed_layer.compute_logits\
            #>> BiasLayer()
        return out.get()
    return maskgit





def nll_loss(y_true, y_pred, sample_weight):
    return tf.reduce_sum(-sample_weight * tf.math.log(tf.gather(tf.nn.softmax(y_pred, -1), y_true, batch_dims=3) + 1e-10)) / (tf.reduce_sum(sample_weight)+1e-10)


MASK_TOKEN = -1
MASKGIT_TRANSFORMER_NAME = "maskgit_transformer"
class MaskGIT(tf.keras.Model):
    def __init__(self, hidden_size, intermediate_size, transformer_heads, transformer_layers, decode_steps, generation_temperature, mask_ratio, *args, **kwargs):
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
        self.mask_ratio = mask_ratio
        input_size = self.input_size_dims[0] * self.input_size_dims[1]
        assert self.input_size_dims[0] == self.input_size_dims[1], "Tokens with different W & H dimensions not supported"



        self._transformer_model = nb.create_model(nb.inp([input_size], dtype=tf.int32), lambda x: x >> create_maskgit(self.codebook_size, hidden_size, input_size, intermediate_size, transformer_heads, transformer_layers), MASKGIT_TRANSFORMER_NAME)
        self._transformer_model.compile(
            tf.optimizers.Adam(),
            nll_loss,
            [tf.metrics.SparseCategoricalAccuracy()]
        )


        self.compile()

        msize = int(round(self.input_size_dims[0] * mask_ratio, 0))
        mask_x = tf.concat([tf.zeros([msize, self.input_size_dims[1]], tf.bool), tf.ones([self.input_size_dims[0] - msize, self.input_size_dims[1]], tf.bool)], 0)
        mask_y = tf.transpose(mask_x)
        inv_x, inv_y = tf.reverse(mask_x, [0]), tf.reverse(mask_y, [1])
        self.train_masks = tf.stack([
            mask_x, mask_y, inv_x, inv_y,
            tf.logical_or(mask_x, mask_y), tf.logical_or(inv_x, mask_y),
            tf.logical_or(mask_x, inv_y), tf.logical_or(inv_x, inv_y)
        ])



    def reshape_to_seq(self, tokens):
        shape = tf.shape(tokens)
        batch, width, height = shape[0], shape[1], shape[2]
        return tf.reshape(tokens, [batch, width*height])

    def reshape_tokens_to_img(self, tokens):
        return tf.reshape(tokens, [tf.shape(tokens)[0], *self.input_size_dims])

    def reshape_logits_to_img(self, logits):
        shape = tf.shape(logits)
        return tf.reshape(logits, [shape[0], *self.input_size_dims, shape[2]])

    # def train_decode(self, input_tokens, target_tokens):
    #     #[batch, positions]
    #     tokens = self.reshape_to_seq(input_tokens)
    #     token_logits = tf.zeros([tf.shape(input_tokens)[0], self.token_count, self.codebook_size])
    #     unknown_counts = tf.reduce_sum(tf.cast(tokens==MASK_TOKEN, tf.int32), -1)
    #     for i in range(self.decode_steps):
    #         with tf.GradientTape() as tape:
    #             write_mask, sampled_tokens, logits = self.decode_step(tokens, unknown_counts, i, training=True)
    #             tokens = tf.where(write_mask, sampled_tokens, tokens)
    #             token_logits = tf.where(write_mask[:, :, tf.newaxis], logits, token_logits)
    #             loss = self._transformer_model.compiled_loss(target_tokens, token_logits, tf.where(write_mask, 1.0, 0.0))
    #         self._transformer_model.optimizer.minimize(loss, self._transformer_model.trainable_variables, tape)
    #     return self.reshape_tokens_to_img(tokens), self.reshape_logits_to_img(token_logits)

    def test_decode(self, input_tokens, decode_steps, generation_temperature):
        #[batch, positions]
        tokens = self.reshape_to_seq(input_tokens)
        token_logits = tf.zeros([tf.shape(input_tokens)[0], self.token_count, self.codebook_size])
        unknown_counts = tf.reduce_sum(tf.cast(tokens==MASK_TOKEN, tf.int32), -1)
        for i in range(decode_steps):
            write_mask, sampled_tokens, logits = self.decode_step(tokens, unknown_counts, i, decode_steps, generation_temperature)
            tokens = tf.where(write_mask, sampled_tokens, tokens)
            token_logits = tf.where(write_mask[:, :, tf.newaxis], logits, token_logits)
        return self.reshape_tokens_to_img(tokens), self.reshape_logits_to_img(token_logits)

    @property
    def token_count(self):
        return self._transformer_model.input_shape[1]

    def mask_schedule(self, mask_ratio):
        return tf.math.cos(np.pi / 2 * mask_ratio)

    def decode_step(self, tokens, initial_unknown_counts, step_i, decode_steps, generation_temperature):
        batch_size = tf.shape(tokens)[0]
        token_count = self.token_count
        #[batch, positions, tokens]
        logits = self(tokens, training=False)

        unknown_tokens = tokens == MASK_TOKEN
        if generation_temperature != 0:
            sampled_tokens = tf.reshape(
                tf.random.categorical(tf.reshape(logits, [batch_size * token_count, self.codebook_size]), num_samples=1, dtype=tf.int32),
                [batch_size, token_count]
            )
        else:
            sampled_tokens = tf.argmax(logits, -1, tf.int32)

        #gather([batch, positions, tokens], [batch, positions]) -> [batch, positions]
        selected_probs = tf.gather(tf.nn.softmax(logits, axis=-1), sampled_tokens, axis=-1, batch_dims=2)
        selected_probs = tf.where(unknown_tokens, selected_probs, np.inf)

        ratio = (step_i + 1.0) / decode_steps
        mask_ratio = self.mask_schedule(ratio)
        #[batch]
        if step_i == decode_steps-1:
            mask_len = tf.zeros_like(initial_unknown_counts)
        else:
            mask_len = tf.maximum(step_i+1, tf.cast(tf.floor(tf.cast(initial_unknown_counts, tf.float32) * mask_ratio), tf.int32))

        confidence = tf.math.log(selected_probs)
        if generation_temperature != 0:
            confidence += tfp.distributions.Gumbel(0, 1).sample(tf.shape(selected_probs)) * generation_temperature * (1 - ratio)

        thresholds = tf.gather(tf.sort(confidence, -1), mask_len[:, tf.newaxis], axis=-1, batch_dims=1)

        write_mask = tf.logical_and(confidence >= thresholds, unknown_tokens)

        return write_mask, sampled_tokens, logits


    def test_decode_simple(self, input_tokens):
        logits = self.reshape_logits_to_img(self(self.reshape_to_seq(input_tokens), training=False))
        return tf.where(input_tokens==MASK_TOKEN, tf.argmax(logits, -1, tf.int32), input_tokens), logits


    def train_decode_simple(self, input_tokens, target_tokens):
        doing_prediction = input_tokens==MASK_TOKEN
        with tf.GradientTape() as tape:
            logits = self.reshape_logits_to_img(self(self.reshape_to_seq(input_tokens), training=True))
            loss = nll_loss(target_tokens, logits, tf.where(doing_prediction, 1.0, 0.0))
        self._transformer_model.optimizer.minimize(loss, self._transformer_model.trainable_variables, tape)
        return tf.where(doing_prediction, tf.argmax(logits, -1, tf.int32), input_tokens), logits


    def call(self, inputs, training=None, mask=None):
        return self._transformer_model(inputs, training=training)
        #return self.decode_simple(inputs, training)[1]
        #return self.decode(inputs, training)[1]


    def train_step(self, data):
        batch_size = tf.shape(data)[0]

        tokens = self._tokenizer.encode(data)

        mask_idx = tf.random.uniform([batch_size], 0, tf.shape(self.train_masks)[0], tf.int32)
        batch_masks = tf.gather(self.train_masks, mask_idx)

        mask_ratios = self.mask_schedule(tf.random.uniform([batch_size, 1, 1]))
        batch_masks = tf.logical_or(tf.random.uniform(tf.shape(tokens)) > mask_ratios, batch_masks)

        train_tokens = tf.where(batch_masks, tokens, MASK_TOKEN)
        sample_weight = tf.where(batch_masks, 0.0, 1.0)

        _, pred_logits = self.train_decode_simple(train_tokens, tokens)

        self._transformer_model.compiled_metrics.update_state(tokens, pred_logits, sample_weight)

        return {
            'main_loss':nll_loss(tokens, pred_logits, sample_weight),
            'learning_rate':self._transformer_model.optimizer.learning_rate
        } | {m.name:m.result() for m in self._transformer_model.metrics}


    def test_step(self, data):
        batch_size = tf.shape(data)[0]

        tokens = self._tokenizer.encode(data)

        mask_idx = tf.random.uniform([batch_size], 0, tf.shape(self.train_masks)[0], tf.int32)
        batch_masks = tf.gather(self.train_masks, mask_idx)
        sample_weights = self.sample_weights(batch_masks)
        _, pred_logits = self.test_decode(self.mask_tokens(tokens, batch_masks), self.decode_steps, self.generation_temp)
        full_loss = nll_loss(tokens, pred_logits, sample_weights)
        full_accuracy = tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(tokens, pred_logits))

        mask_ratios = self.mask_schedule(tf.random.uniform([batch_size, 1, 1]))
        batch_masks_masked = tf.logical_or(tf.random.uniform(tf.shape(tokens)) > mask_ratios, batch_masks)
        sample_weights = self.sample_weights(batch_masks_masked)
        _, pred_logits = self.test_decode_simple(self.mask_tokens(tokens, batch_masks_masked))
        simple_loss = nll_loss(tokens, pred_logits, sample_weights)
        simple_accuracy = tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(tokens, pred_logits))

        return {
            "full_loss": full_loss,
            "full_accuracy": full_accuracy,
            "simple_loss": simple_loss,
            "simple_accuracy": simple_accuracy
        }


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
        return self.from_tokens(self.test_decode_simple(tokens)[0], training)


    def decode_img(self, tokens, decode_steps, generation_temperature, training = False):
        return self.from_tokens(self.test_decode(tokens, decode_steps, generation_temperature)[0], training)

    @staticmethod
    def mask_tokens(tokens, mask):
        return tf.where(mask, tokens, MASK_TOKEN)

    @staticmethod
    def sample_weights(mask):
        return tf.where(mask, 0.0, 1.0)

    @staticmethod
    def load(dirname) -> "MaskGIT":
        return tf.keras.models.load_model(dirname, custom_objects={
            "MaskGIT":MaskGIT, "VQVAEModel":VQVAEModel, "BiasLayer":BiasLayer, "TokenEmbedding":TokenEmbedding
            #"TransformerLayer":TransformerLayer, "TransformerMLP":TransformerMLP, "TransformerAttention":TransformerAttention
        })

    def set_generation_temp(self, temp):
        self.generation_temperature = temp

    @staticmethod
    def new(args):
        return MaskGIT(args.hidden_size, args.intermediate_size, args.transformer_heads, args.transformer_layers, args.decode_steps, args.generation_temperature, args.mask_ratio)

    def get_config(self):
        return super().get_config() | {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "transformer_heads": self.transformer_heads,
            "transformer_layers": self.transformer_layers,
            "decode_steps": self.decode_steps,
            "generation_temperature": self.generation_temperature,
            "mask_ratio": self.mask_ratio
        }



def main(args):
    tf_init(args.use_gpu, args.threads, args.seed)

    class MaskGITLog(TrainingLog):
        def run_test(self, data):
            assert isinstance(self.model, MaskGIT), "This callback can only be used with MaskGIT models"

            tokens_original = self.model.to_tokens(data)

            masks = self.model.get_masks(tf.shape(data)[0])
            tokens_masked = tf.where(masks, tokens_original, MASK_TOKEN)

            recreated = self.model.from_tokens(tokens_original)

            reconstructed_simple = self.model.decode_simple_img(tokens_masked)
            reconstructed = self.model.decode_img(tokens_masked, decode_steps=12)
            generated = self.model.decode_img(MASK_TOKEN * tf.ones_like(tokens_original), decode_steps=12)
            self.log.log_images("images", data).log_images("tokenizer_recreated", recreated).log_images("reconstruction_simple", reconstructed_simple)\
                .log_images("reconstruction", reconstructed).log_images("generated_images", generated)

    if not args.load_model:
        model = MaskGIT.new(args)
    else:
        model = MaskGIT.load(get_maskgit_fname())

    image_load = ImageLoading(args.dataset_location, args.img_size, args.places)

    train_dataset, dev_dataset = image_load.create_train_dev_datasets(1000, args.batch_size)
    show_dataset = image_load.create_dataset(8)


    run_name = WandbManager("image_outpainting_maskgit").start(args)
    model_log = MaskGITLog(dev_dataset, show_dataset, 25, 400)


    callbacks = [
        WandbModelCheckpoint(get_maskgit_fname(run_name), "main_loss", save_freq=10000),
        model_log
    ]
    #model.pretraining()
    #model.fit(train_dataset, epochs=1, callbacks=callbacks)
    #model.normal_train()
    model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks)


if __name__ == "__main__":
    _given_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(_given_args)
