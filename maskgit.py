import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from wandb.keras import WandbModelCheckpoint


from log_and_save import WandbLog, TrainingLog
from dataset import ImageLoading
import tokenizer
from tokenizer import VQVAEModel
from utilities import get_tokenizer_fname, get_maskgit_fname
from tf_utilities import tf_init
from build_network import NetworkBuild as nb, NBTensor


parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--hidden_size", default=256, type=int, help="MaskGIT transformer hidden size")
parser.add_argument("--intermediate_size", default=512, type=int, help="MaskGIT transformer intermediate size")
parser.add_argument("--transformer_heads", default=4, type=int, help="MaskGIT transformer heads")
parser.add_argument("--transformer_layers", default=2, type=int, help="MaskGIT transformer layers (how many times do we repeat the self attention block and MLP)")
parser.add_argument("--decode_steps", default=12, type=int, help="MaskGIT decoding steps")
parser.add_argument("--generation_temperature", default=1.0, type=float, help="How random is MaskGIT during decoding")
parser.add_argument("--mask_ratio", default=0.5, type=float, help="Percentage of tokens to mask during training")



parser.add_argument("--dataset_location", default="", type=str, help="Directory to read data from. If not set, the path in the environment variable IMAGE_OUTPAINTING_DATASET_LOCATION is used instead.")
parser.add_argument("--load_model", default=False, type=bool, help="Whether to load a maskGIT model")





class TokenEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer for MaskGIT - token embedding + positional embedding
    """
    def __init__(self, codebook_size, position_count, embedding_dim, **kwargs):
        super().__init__(**kwargs)

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        #the embeddings for all tokens
        self.codebook = self.add_weight("codebook", [codebook_size, embedding_dim], tf.float32, tf.keras.initializers.TruncatedNormal(0, 0.02))

        #positional embeddings for every position
        self.positional_embedding = self.add_weight("position_embed", [position_count, embedding_dim], tf.float32, tf.keras.initializers.TruncatedNormal())

    def compute_logits(self, values):
        """Compute dot product between each value and each vector in the codebook. Can be used as model output"""
        #dimensions - [batch, seq_len, embed_dim] @ [codebook_size, embed_dim]^T  ->  [batch, seq_len, codebook_size]
        return values @ tf.transpose(self.codebook)

    def call(self, inputs, *args, **kwargs):
        """Replace all legitimate tokens with their embeddings, and all MASK_TOKENS with 0"""
        mask = inputs==MASK_TOKEN
        #all tokens to their embeddings, MASK_TOKENS to vectors of zeroes
        tokens_embed = tf.where(mask[:, :, tf.newaxis], 0.0, tf.gather(self.codebook, tf.where(mask, 0, inputs), axis=0))
        return tokens_embed + self.positional_embedding


class BiasLayer(tf.keras.layers.Layer):
    """Bias for every component along the last dimension."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias = None

    def build(self, input_shape):
        #create bias with the same shape as the last input dimension
        self.bias = self.add_weight('bias', shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, inputs, *args, **kwargs):
        """Add the trainable bias to the inputs"""
        return inputs + self.bias


def create_maskgit(codebook_size, hidden_size, position_count, intermediate_size, transformer_heads, transformer_layers):
    """Create the MaskGIT model"""
    def maskgit(tokens):
        #embed the tokens
        embed_layer = TokenEmbedding(codebook_size, position_count, hidden_size)

        #compute embeddings - token embeds + position embeds
        embed = embed_layer(tokens)
        #normalize the embedding and add dropout
        x = NBTensor(embed) >> nb.layer_norm() >> nb.dropout(0.1)
        #add all transformer layers
        for _ in range(transformer_layers):
            x = x >> nb.transformer_layer(hidden_size, intermediate_size, transformer_heads)

        #add a dense processing layer, a layer normalization, and predict the logits over all tokens
        out = x\
            >> nb.dense(hidden_size, activation="gelu")\
            >> nb.layer_norm()\
            >> nb.dense(codebook_size, activation=None)
            #>> embed_layer.compute_logits\
            #>> BiasLayer()
        return out.get()
    return maskgit





# def nll_loss(y_true, y_pred, sample_weight):
#     return tf.reduce_sum(-sample_weight * tf.math.log(tf.gather(tf.nn.softmax(y_pred, -1), y_true, batch_dims=3) + 1e-10)) / (tf.reduce_sum(sample_weight)+1e-10)


MASK_TOKEN = -1
MASKGIT_TRANSFORMER_NAME = "maskgit_transformer"
class MaskGIT(tf.keras.Model):
    """The MaskGIT model class"""
    def __init__(self, hidden_size, intermediate_size, transformer_heads, transformer_layers, decode_steps, generation_temperature, mask_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #load the tokenizer to use during training
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


        #create the transformer model - a downscaled version of the model from the MaskGIT article. Takes tokens as input and produces probabilities of masked tokens on output
        self._transformer_model = nb.create_model(nb.inp([input_size], dtype=tf.int32), lambda x: x >> create_maskgit(self.codebook_size, hidden_size, input_size, intermediate_size, transformer_heads, transformer_layers), MASKGIT_TRANSFORMER_NAME)
        self._transformer_model.compile(
            tf.optimizers.Adam(),
            tf.losses.SparseCategoricalCrossentropy(),
            [tf.metrics.SparseCategoricalAccuracy()]
        )

        self.compile()

        #prepare masks for outpainting - unlike the original article, we always start with a part of an image and try to reconstruct the rest
        #mask size - how many tokens should be removed during training
        mask_size = int(round(self.input_size_dims[0] * mask_ratio, 0))
        #construct x mask - as large as tokens, zeroes on the left representing masked tokens, ones on the right
        mask_x = tf.concat([tf.zeros([mask_size, self.input_size_dims[1]], tf.bool), tf.ones([self.input_size_dims[0] - mask_size, self.input_size_dims[1]], tf.bool)], 0)
        #y mask - zeroes on the top, ones at the bottom
        mask_y = tf.transpose(mask_x)
        #inverse masks - reverse x/y masks
        inv_x, inv_y = tf.reverse(mask_x, [0]), tf.reverse(mask_y, [1])
        # train masks - for each training example, one of them will be picked, the tokens with 0 will be masked out and reconstruction will happen
        # we are either filling in a side (mask_x, mask_y, inv_x, inv_y), or a corner = logical or of two other masks
        self.train_masks = tf.stack([
            mask_x, mask_y, inv_x, inv_y,
            tf.logical_or(mask_x, mask_y), tf.logical_or(inv_x, mask_y),
            tf.logical_or(mask_x, inv_y), tf.logical_or(inv_x, inv_y)
        ])



    def _reshape_to_seq(self, tokens):
        """Reshape tokens of size [batch, width, height] to size [batch, width*height]"""
        shape = tf.shape(tokens)
        batch, width, height = shape[0], shape[1], shape[2]
        return tf.reshape(tokens, [batch, width*height])

    def _reshape_tokens_to_img(self, tokens):
        """Reshape tokens of size [batch, width*height] to size [batch, width, height]"""
        return tf.reshape(tokens, [tf.shape(tokens)[0], *self.input_size_dims])

    def _reshape_logits_to_img(self, logits):
        """Reshape tokens of size [batch, width*height, token_count] to size [batch, width, height, token_count]"""
        shape = tf.shape(logits)
        return tf.reshape(logits, [shape[0], *self.input_size_dims, shape[2]])

    def test_decode(self, input_tokens, decode_steps, generation_temperature):
        """Use the model to generate masked tokens using multiple steps and given generation temperature"""
        tokens = self._reshape_to_seq(input_tokens)
        #final token logits
        token_logits = tf.zeros([tf.shape(input_tokens)[0], self.token_count, self.codebook_size])
        #how many masked tokens are present in each input
        unknown_counts = tf.reduce_sum(tf.cast(tokens==MASK_TOKEN, tf.int32), -1)
        #go over all decode steps
        for i in range(decode_steps):
            #do a decode steps - make a guess at a few tokens
            write_mask, sampled_tokens, logits = self._decode_step(tokens, unknown_counts, i, decode_steps, generation_temperature)
            #save the tokens and logits generated during the current step
            tokens = tf.where(write_mask, sampled_tokens, tokens)
            token_logits = tf.where(write_mask[:, :, tf.newaxis], logits, token_logits)
        return self._reshape_tokens_to_img(tokens), self._reshape_logits_to_img(token_logits)

    @property
    def token_count(self):
        """The length of the token sequence used as transformer input"""
        return self._transformer_model.input_shape[1]

    def _mask_schedule(self, mask_ratio):
        """Compute the fraction of masked tokens according to the cosine schedule from the MaskGIT article"""
        return tf.math.cos(np.pi / 2 * mask_ratio)

    def _decode_step(self, tokens, initial_unknown_counts, step_i, decode_steps, generation_temperature):
        """Perform one decoding step - figure out the most probable tokens to write during this step"""
        batch_size = tf.shape(tokens)[0]
        token_count = self.token_count

        #make the model predict the logits of all tokens for all masked positions
        logits = self(tokens, training=False)

        unknown_tokens = tokens == MASK_TOKEN

        #if I should generate tokens a bit randomly, sample tokens according to their probabilitiesz
        if generation_temperature != 0:
            #sample each token from the categorical distribution defined by the logits for each position
            sampled_tokens = tf.reshape(
                tf.random.categorical(tf.reshape(logits, [batch_size * token_count, self.codebook_size]), num_samples=1, dtype=tf.int32),
                [batch_size, token_count]
            )
        else:
            #just take the token that is the most probablie
            sampled_tokens = tf.argmax(logits, -1, tf.int32)

        #get the probabilities of the tokens we sampled
        selected_probs = tf.gather(tf.nn.softmax(logits, axis=-1), sampled_tokens, axis=-1, batch_dims=2)
        #set the probabilities of tokens that aren't masked to infinity (we are entirely sure they are correct)
        selected_probs = tf.where(unknown_tokens, selected_probs, np.inf)

        #how far in the generation process are we
        ratio = (step_i + 1.0) / decode_steps
        #what part of the tokens should be masked after this step finishes
        mask_ratio = self._mask_schedule(ratio)

        #if this is the last step, we need to generate all the remaining tokens - 0 will remain masked
        if step_i == decode_steps-1:
            mask_len = tf.zeros_like(initial_unknown_counts)
        else:
            #else, generate at least one token during this step, or behave according to the generation schedule
            mask_len = tf.maximum(step_i+1, tf.cast(tf.floor(tf.cast(initial_unknown_counts, tf.float32) * mask_ratio), tf.int32))

        #convert selected probabilities back to logits
        confidence = tf.math.log(selected_probs)
        #add random noise to all logits to make the process a bit random
        if generation_temperature != 0:
            confidence += tfp.distributions.Gumbel(0, 1).sample(tf.shape(selected_probs)) * generation_temperature * (1 - ratio)

        #figure out the (mask_len)-th smallest confidence - all confidences below this will be masked, we are ready to predict the rest
        thresholds = tf.gather(tf.sort(confidence, -1), mask_len[:, tf.newaxis], axis=-1, batch_dims=1)

        #we write the tokens we don't know yet and whose confidence is above the threshold
        write_mask = tf.logical_and(confidence >= thresholds, unknown_tokens)

        return write_mask, sampled_tokens, logits


    def test_decode_simple(self, input_tokens):
        """Run the internal model once, return the most probable tokens and their logits"""
        logits = self._reshape_logits_to_img(self(self._reshape_to_seq(input_tokens), training=False))
        return tf.where(input_tokens==MASK_TOKEN, tf.argmax(logits, -1, tf.int32), input_tokens), logits


    def train_decode_simple(self, input_tokens, target_tokens):
        """Train the model by decoding given input tokens and trying to get target_tokens"""
        doing_prediction = input_tokens==MASK_TOKEN
        with tf.GradientTape() as tape:
            #predict the logits of the target tokens
            logits = self._reshape_logits_to_img(self(self._reshape_to_seq(input_tokens), training=True))
            #compute the loss based on predicted tokens
            loss = self._transformer_model.compiled_loss(target_tokens, logits, tf.where(doing_prediction, 1.0, 0.0))
        #minimize the loss
        self._transformer_model.optimizer.minimize(loss, self._transformer_model.trainable_variables, tape)
        return tf.where(doing_prediction, tf.argmax(logits, -1, tf.int32), input_tokens), logits


    def call(self, inputs, training=None, mask=None):
        """Call the transformer model on the given inputs"""
        return self._transformer_model(inputs, training=training)

    def train_step(self, data):
        """Run a training step on the given batch of images"""
        batch_size = tf.shape(data)[0]

        #convert images to tokens
        tokens = self._tokenizer.encode(data)

        #randomly select the training masks to use for each example
        mask_idx = tf.random.uniform([batch_size], 0, tf.shape(self.train_masks)[0], tf.int32)
        batch_masks = tf.gather(self.train_masks, mask_idx)

        #figure out the ratio of tokens to mask in addition to the mask selected above
        mask_ratios = self._mask_schedule(tf.random.uniform([batch_size, 1, 1]))
        #mask out approximately the above specified number of tokens
        batch_masks = tf.logical_or(tf.random.uniform(tf.shape(tokens)) > mask_ratios, batch_masks)

        #the tokens from which we will try to predict the original - replace some with MASK_TOKEN
        train_tokens = self._mask_tokens(tokens, batch_masks)

        #train the model to predict tokens based on train_tokens
        _, pred_logits = self.train_decode_simple(train_tokens, tokens)

        #we only compute loss and metrics on the masked tokens
        sample_weight = self._mask_sample_weights(batch_masks)

        #update metrics
        self._transformer_model.compiled_metrics.update_state(tokens, pred_logits, sample_weight)

        #return model losses and metrics
        return {
            'main_loss':self._transformer_model.compiled_loss(tokens, pred_logits, sample_weight),
            'learning_rate':self._transformer_model.optimizer.learning_rate
        } | {m.name:m.result() for m in self._transformer_model.metrics}


    def test_step(self, data):
        """Perform a test step - try decoding data both during one step and during multiple ones, measure the accuracy in both cases"""
        batch_size = tf.shape(data)[0]

        #convert image to tokens
        tokens = self._tokenizer.encode(data)

        #get random masks to start from
        mask_idx = tf.random.uniform([batch_size], 0, tf.shape(self.train_masks)[0], tf.int32)
        batch_masks = tf.gather(self.train_masks, mask_idx)

        #get sample weights from batch masks
        sample_weights = self._mask_sample_weights(batch_masks)

        #compute logits using the full decoding with no randomness, measure loss and accuracy
        _, pred_logits_full = self.test_decode(self._mask_tokens(tokens, batch_masks), self.decode_steps, 0.0)
        full_loss = self._transformer_model.compiled_loss(tokens, pred_logits_full, sample_weights)
        full_accuracy = tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(tokens, pred_logits_full))

        #compute logits using the simple decoding, measure loss and accuracy
        _, pred_logits_simple = self.test_decode_simple(self._mask_tokens(tokens, batch_masks))
        simple_loss = self._transformer_model.compiled_loss(tokens, pred_logits_simple, sample_weights)
        simple_accuracy = tf.reduce_mean(tf.metrics.sparse_categorical_accuracy(tokens, pred_logits_simple))

        #return the resulting values of loss and accuracy
        return {
            "full_loss": full_loss,
            "full_accuracy": full_accuracy,
            "simple_loss": simple_loss,
            "simple_accuracy": simple_accuracy
        }


    @property
    def downscale_multiplier(self):
        """How much does tokenizer downscale images. Equal to pixels per side of one token"""
        return self._tokenizer.downscale_multiplier


    def get_masks(self, data_len):
        """Get the given number of masks, repeating the order in which they were created"""
        mask_idx = tf.range(data_len) % tf.shape(self.train_masks)[0]
        return tf.gather(self.train_masks, mask_idx)


    def to_tokens(self, img, training=False):
        """Convert given image to tokens"""
        return self._tokenizer.encode(img, training)

    def from_tokens(self, tokens, training=False):
        """Convert given tokens back to an image"""
        return self._tokenizer.decode(tokens, training)

    def decode_simple_img(self, tokens):
        """Use the model to predict masked tokens using the simple method"""
        return self.from_tokens(self.test_decode_simple(tokens)[0])

    def decode_img(self, tokens, decode_steps, generation_temperature):
        """Use the model to predict masked tokens using multiple decoding steps"""
        return self.from_tokens(self.test_decode(tokens, decode_steps, generation_temperature)[0])

    @staticmethod
    def _mask_tokens(tokens, mask):
        """Mask tokens from `tokens` according to the provided mask"""
        return tf.where(mask, tokens, MASK_TOKEN)

    @staticmethod
    def _mask_sample_weights(mask):
        """1.0 for all places with masked tokens, 0.0 elsewhere"""
        return tf.where(mask, 0.0, 1.0)

    @staticmethod
    def load(dirname) -> "MaskGIT":
        """Load MaskGIT from a filename"""
        return tf.keras.models.load_model(dirname, custom_objects={
            "MaskGIT":MaskGIT, "VQVAEModel":VQVAEModel, "BiasLayer":BiasLayer, "TokenEmbedding":TokenEmbedding
            #"TransformerLayer":TransformerLayer, "TransformerMLP":TransformerMLP, "TransformerAttention":TransformerAttention
        })

    @staticmethod
    def new(args : argparse.Namespace):
        """Create a new MaskGIT instance using the provided arguments"""
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
    #initialize the GPU to use, available CPU threads, and seed
    tf_init(args.use_gpu, args.threads, args.seed)

    class MaskGITLog(TrainingLog):
        """Used to log progress of MaskGIT training"""

        def run_test(self, data):
            assert isinstance(self.model, MaskGIT), "This callback can only be used with MaskGIT models"

            #convert images to tokens
            tokens_original = self.model.to_tokens(data)
            #just convert the tokens back instantly
            recreated = self.model.from_tokens(tokens_original)

            #mask out some tokens and let the model recreate the tokens
            masks = self.model.get_masks(tf.shape(data)[0])
            tokens_masked = tf.where(masks, tokens_original, MASK_TOKEN)

            #let the model reconstruct the masked tokens using the simple method
            reconstructed_simple = self.model.decode_simple_img(tokens_masked)
            #let the model reconstruct the masked tokens using the multi-step method
            reconstructed = self.model.decode_img(tokens_masked, decode_steps=12, generation_temperature=1.0)

            #let the model generate some images. Since it is not trained on this task, the quality will be subpar, but it is interesting nonetheless
            generated = self.model.decode_img(MASK_TOKEN * tf.ones_like(tokens_original), decode_steps=12, generation_temperature=1.0)
            #log all created images to wandb
            self.log.log_images("images", data).log_images("tokenizer_recreated", recreated).log_images("reconstruction_simple", reconstructed_simple)\
                .log_images("reconstruction", reconstructed).log_images("generated_images", generated)

    #load the MaskGIT model if requested
    if not args.load_model:
        model = MaskGIT.new(args)
    else:
        model = MaskGIT.load(get_maskgit_fname())

    #create the train, dev and show datasets
    image_load = ImageLoading(args.dataset_location, args.img_size)
    train_dataset, dev_dataset = image_load.create_train_dev_datasets(1000, args.batch_size)
    show_dataset = image_load.create_dataset(8)

    #start wandb logging
    run_name = WandbLog.wandb_init("image_outpainting_maskgit", args)
    model_log = MaskGITLog(dev_dataset, show_dataset, 25, 400)

    #train the model
    model.fit(train_dataset, epochs=args.epochs, callbacks = [
        WandbModelCheckpoint(get_maskgit_fname(run_name), "main_loss", save_freq=10000),
        model_log
    ])


if __name__ == "__main__":
    _args = parser.parse_args([] if "__file__" not in globals() else None)
    main(_args)
