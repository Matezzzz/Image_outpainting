import argparse

import numpy as np
import tensorflow as tf
import wandb

from build_network import NetworkBuild as nb, NBTensor
from dataset import ImageLoading
from log_and_save import TrainingLog, WandbLog
from wandb.keras import WandbModelCheckpoint
from utilities import get_tokenizer_fname
from tf_utilities import tf_init


parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--filters", default=32, type=int, help="Base residual layer filters")
parser.add_argument("--residual_layer_multipliers", default=[1, 2, 3], nargs="+", type=int, help="How many residual layers to use and how to increase number of filters")
parser.add_argument("--num_res_blocks", default=2, type=int, help="Number of residual blocks in sequence before a downscale")
parser.add_argument("--embedding_dim", default=32, type=int, help="Embedding dimension for quantizer")
parser.add_argument("--codebook_size", default=128, type=int, help="Codebook size for quantizer")
parser.add_argument("--embedding_loss_weight", default=1, type=float, help="Scale the embedding loss")
parser.add_argument("--commitment_loss_weight", default=0.25, type=float, help="Scale for the commitment loss (penalize changing embeddings)")
parser.add_argument("--entropy_loss_weight", default=0.01, type=float, help="Scale for the entropy loss")
parser.add_argument("--discriminator_loss_weight", default=0.0, type=float, help="Scale for the discriminator loss")
parser.add_argument("--decoder_noise_dim", default=0, type=int, help="How many dimensions of loss to add before decoding")


parser.add_argument("--dataset_location", default="", type=str, help="Directory to read data from. If not set, the path in the environment variable IMAGE_OUTPAINTING_DATASET_LOCATION is used instead.")
parser.add_argument("--load_model", default=False, type=bool, help="Whether to load model or not")



class VectorQuantizer(tf.keras.layers.Layer):
    """
    The VectorQuantizer class - performs the vector quantization operation from VQVAE, implements the respective losses, and enables conversions from tokens to embedding space and back
    """
    def __init__(self, codebook_size, embedding_dim, embedding_loss_weight, commitment_loss_weight, entropy_loss_weight, **kwargs):
        """
        Create a new VectorQuantizer layer with the given parameters
        
        Args
        ----
        codebook_size - how many vectors should this layer use
        embedding_dim - dimensionality of all codebook vectors
        embedding_loss_weight - multiplier of the embedding loss (moving model predictions closer to embed vectors)
        commitment_loss_weight - multiplier of the commitment loss (moving embedding vectors closer to model predictions)
        entropy_loss_weight - multiplier of the entropy loss (moving unused embedding vectors to the closest model prediction)
        """
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embedding_loss_weight = embedding_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        #create the codebook, initialize it randomly
        self.codebook = self.add_weight("codebook", [codebook_size, embedding_dim], tf.float32, tf.keras.initializers.RandomUniform())

    @staticmethod
    def new(args : argparse.Namespace):
        """
        Create a new VectorQuantizer layer based on the given arguments
        """
        return VectorQuantizer(
            args.codebook_size,
            args.embedding_dim,
            args.embedding_loss_weight,
            args.commitment_loss_weight,
            args.entropy_loss_weight
        )


    def get_embedding(self, tokens):
        """
        Convert tokens to embedding space
        """
        return tf.gather(self.codebook, tokens, axis=0)

    def get_distances(self, values):
        """
        Get L2 distances of values of shape [batch, width, height, embed] to each embedding vectors
        """
        #values[:, :, :, tf.newaxis, :] - self.codebook -- shapes [batch, w, h, 1, embed] - [codebook_size, embed]
        #Gets differences between each pair of values and codebook vectors
        #distance is then $diff^T * diff$ for each difference. Is computed as reduce_sum(square(diff)) here
        return tf.reduce_sum(tf.square(values[:, :, :, tf.newaxis, :] - self.codebook), -1)

    def get_tokens(self, distances):
        """
        Get tokens when distances are known (return the closest token for each value)
        """
        return tf.argmin(distances, -1, tf.int32)


    def call(self, inputs, *args, training=None, **kwargs):
        """
        Call the layer - convert inputs to quantized tokens and back, compute all losses
        """
        #compute distances to all codebook vectors
        distances = self.get_distances(inputs)
        #get discrete tokens
        tokens = self.get_tokens(distances)
        #convert back into embedding space
        embeddings = self.get_embedding(tokens)

        #embedding loss - minimize distance between predictions and tokens (used to train the encoder, embeddings are unaffected)
        embedding_loss = self.embedding_loss_weight * tf.reduce_mean((embeddings - tf.stop_gradient(inputs))**2)
        #commitment loss - minimize distance between predictions and tokens (trains the embeddings)
        commitment_loss = self.commitment_loss_weight * tf.reduce_mean((tf.stop_gradient(embeddings) - inputs)**2)

        #Old entropy loss (not used now) - moved all embeddings to their closest prediction
        #entropy_loss = self.entropy_loss_weight * tf.reduce_mean(tf.reduce_min(tf.reshape(distances, [-1, self.codebook_size]), 0)) #self.compute_entropy_loss(distances)

        #New entropy loss - if an embedding has not been used this batch, move it to the closest vector
        flat_distances = tf.reshape(distances, [-1, self.codebook_size])
        # get the closest class for each token in the current batch
        closest_classes = tf.argmin(flat_distances, 1, tf.int32)
        # compute the amounts of tokens present in each class
        active_classes = tf.scatter_nd(closest_classes[:, tf.newaxis], tf.ones_like(closest_classes), [self.codebook_size])
        # if the amount of tokens in a class is 0, minimize the smallest distance (move it to the closest prediction)
        entropy_loss = self.entropy_loss_weight * tf.reduce_mean(tf.where(active_classes == 0, tf.reduce_min(flat_distances, 0), 0.0), 0)

        #add all three losses
        self.add_loss(commitment_loss + embedding_loss + entropy_loss)
        #add metrics for each loss
        self.add_metric(commitment_loss, "commitment_loss")
        self.add_metric(embedding_loss, "embedding_loss")
        self.add_metric(entropy_loss, "entropy_loss")

        #if training, propagate the unmodified gradient to the output
        if training:
            embeddings = inputs + tf.stop_gradient(embeddings - inputs)
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "codebook_size":self.codebook_size,
            "embedding_dim":self.embedding_dim,
            "embedding_loss_weight":self.embedding_loss_weight,
            "commitment_loss_weight":self.commitment_loss_weight,
            "entropy_loss_weight":self.entropy_loss_weight
        })
        return config




def create_encoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim):
    #create the encoder - start with downscaling residual blocks, then group norm, swish and a 2d convolution predicting the vectors in the embedding space
    def encoder(x):
        out = NBTensor(x)\
            >> nb.residual_downscale_sequence(filters * np.array(residual_layer_multipliers), num_res_blocks, nb.batch_norm, nb.swish)\
            >> nb.group_norm()\
            >> nb.swish\
            >> nb.conv_2d(embedding_dim, kernel_size=1)
        return out.get()
    return encoder




def create_decoder(filters, residual_layer_multipliers, num_res_blocks, decoder_noise_dim):
    #create the decoder
    def decoder(x):
        #when decoding, we might want to add some noise dimensions to the input (similar to the random noise used as input for GANs)
        #this is useful only when a discriminator is used (args.discriminator_loss_weight != 0)
        def add_noise_dims(x):
            #x = [batch, tok_x, tok_y, channels]
            if decoder_noise_dim > 0:
                shape = tf.shape(x)
                return tf.concat([x, tf.random.normal([shape[0], shape[1], shape[2], decoder_noise_dim])], -1)
            return x
        # add noise, one processing convolution, then upscale using tranposed convolutions and residual blocks. Do group norm, swish, then predict the final colors
        out = NBTensor(x)\
            >> add_noise_dims\
            >> nb.conv_2d(filters * residual_layer_multipliers[-1])\
            >> nb.residual_upscale_sequence(filters * np.array(residual_layer_multipliers[::-1]), num_res_blocks, nb.batch_norm, nb.swish)\
            >> nb.group_norm()\
            >> nb.swish\
            >> nb.conv_2d(filters=3)
        return out.get()
    return decoder



def create_discriminator(image_size, residual_blocks):
    """Create the discriminator model"""
    def discriminator(x : NBTensor):
        """Downscaling convolution -> Multiple downscaling residual blocks -> global average pooling -> dense prediction layer"""
        y = x\
            >> nb.conv_2d_down(8, activation="relu")\
            >> nb.residual_downscale_sequence([8, 16, 32, 64, 128], residual_blocks)\
            >> nb.conv_2d(128, activation="relu")\
            >> nb.global_average_pooling_2d()\
            >> nb.dense(2, activation="softmax")
        return y
    model = nb.create_model(nb.inp([image_size, image_size, 3]), discriminator)
    model.compile(
        tf.optimizers.Adam(),
        tf.losses.SparseCategoricalCrossentropy(),
        tf.metrics.SparseCategoricalAccuracy(name="discriminator_accuracy")
    )
    return model



#image data variance - used to scale the reconstruction loss, as in the original VQVAE article
DATA_VARIANCE = 0.0915055541062209
def scaled_mse_loss(y_true, y_pred):
    return tf.reduce_mean((y_true-y_pred)**2) / DATA_VARIANCE


# pylint: disable=abstract-method
class VQVAEModel(tf.keras.Model):
    """
    The VQVAE model - contains an encoder, decoder and the vector quantization layer. Enables converting images from and to tokens.
    """
    def __init__(self, img_size, filters, residual_layer_multipliers, num_res_blocks, embedding_dim, codebook_size, decoder_noise_dim,
            embedding_loss_weight, commitment_loss_weight, entropy_loss_weight, discriminator_loss_weight, *args, **kwargs):
        """
        Create a VQVAE model with the given parameters
        """

        # a layer that performs the vector quantization operation - contains all embedding vectors
        _quantizer = VectorQuantizer(codebook_size, embedding_dim, embedding_loss_weight, commitment_loss_weight, entropy_loss_weight)
        self.img_size = img_size
        self.filters = filters
        self.residual_layer_multipliers = residual_layer_multipliers
        self.num_res_blocks = num_res_blocks
        self.discriminator_loss_weight = discriminator_loss_weight
        self.decoder_noise_dim = decoder_noise_dim

        inp_shape = [img_size, img_size, 3]

        #create an encoding and a decoding model
        _encode_model = nb.create_model(nb.inp(inp_shape),
                                             lambda x: x >> create_encoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim))

        _decode_model = nb.create_model(nb.inp(_encode_model.output_shape[1:]),
                                             lambda x: x >> create_decoder(filters, residual_layer_multipliers, num_res_blocks, decoder_noise_dim))

        #the model that will be trained - pass through the encoder, then the quantization operation, then the decoder
        inp, out = nb.create_model_io(nb.inp(inp_shape), lambda x: x >> _encode_model >> _quantizer >> _decode_model)

        super().__init__(inp, out, *args, **kwargs)
        self._encode_model, self._quantizer, self._decode_model = _encode_model, _quantizer, _decode_model

        self.compile(
            tf.optimizers.Adam(),
            scaled_mse_loss
        )

        #if the discriminator should be used, create it
        if self.discriminator_loss_weight != 0:
            self._discriminator = create_discriminator(img_size, 2)


    def train_step(self, data):
        """
        Perform a single training step on the given data
        """
        #all tracked metrics
        metrics = {}

        with tf.GradientTape() as tape:
            #let the model reconstruct the data
            reconstruction = self(data, training=True)

            batch_size = tf.shape(data)[0]
            #if a discriminator is supposed to be used
            if self.discriminator_loss_weight != 0:
                #call the discriminator on the created images. The loss = how much the discriminator believes the image to be fake
                disc_out = self._discriminator(reconstruction, training=True)
                discriminator_loss = self.discriminator_loss_weight * self._discriminator.compiled_loss(tf.ones(batch_size), disc_out)
                metrics |= {'discriminator_loss':discriminator_loss}
            else:
                discriminator_loss = 0

            #reconstruction loss = just MSE
            reconstruction_loss = self.compiled_loss(data, reconstruction)
            #vqvae loss - sum of commitment loss, reconstruction loss, embedding loss and any regularization losses from any of the model layers
            vqvae_loss = sum(self.losses)

            #compute the total loss
            loss = reconstruction_loss + discriminator_loss + vqvae_loss
            #save all metrics
            metrics |= {'main_loss':loss, 'reconstruction_loss': reconstruction_loss, 'vqvae_loss':vqvae_loss, 'learning_rate':self.optimizer.learning_rate}
            #update all model metrics states
            self.compiled_metrics.update_state(data, reconstruction)
            metrics |= {m.name:m.result() for m in self.metrics}

        #train the model by minimizing loss with respect to all variables
        self.optimizer.minimize(loss, self.trainable_variables, tape)

        #train the discriminator if it is being used
        if self.discriminator_loss_weight != 0:
            with tf.GradientTape() as tape:
                #create the training data and labels. (fake = label 0, real image = label 1)
                disc_labels = tf.concat([tf.zeros(batch_size), tf.ones(batch_size)], 0)
                disc_input = tf.concat([reconstruction, data], 0)

                #call the discriminator on the data and compute the loss
                disc_predictions = self._discriminator(disc_input, training=True)

                disc_loss = self._discriminator.compiled_loss(disc_labels, disc_predictions)
            #train the discriminator using its' loss
            self._discriminator.optimizer.minimize(disc_loss, self._discriminator.trainable_variables, tape)
            self._discriminator.compiled_metrics.update_state(disc_labels, disc_predictions)
            #save all discriminator metrics
            metrics |= {'D_discriminator_loss': disc_loss} | {("D_"+m.name):m.result() for m in self._discriminator.metrics}
        return metrics

    def generate_samples(self, count):
        """
        Generate random tokens, then use the decoder to create an image from them
        """
        #create random tokens of shape [count, token_image_width, token_image_height]
        tokens = np.random.randint(0, self.codebook_size, [count, self._encode_model.output_shape[1], self._encode_model.output_shape[2]])
        #convert tokens into embedding space, then call the decoder on them
        embed_tokens = self._quantizer.get_embedding(tokens)
        return self._decode_model.predict(embed_tokens, verbose=0)

    @staticmethod
    def new(args : argparse.Namespace):
        """
        Create a new VQVAE model using argparse arguments.
        """
        return VQVAEModel(args.img_size, args.filters, args.residual_layer_multipliers, args.num_res_blocks, args.embedding_dim, args.codebook_size, args.decoder_noise_dim,
                          args.embedding_loss_weight, args.commitment_loss_weight, args.entropy_loss_weight, args.discriminator_loss_weight)

    @staticmethod
    def load(fname) -> "VQVAEModel":
        """
        Load a VQVAE model from the given filename / directory
        """



        #load the model
        return tf.keras.models.load_model(fname, VQVAEModel.custom_objects(), compile=False)


    @property
    def input_shape(self):
        """
        Input shape - image width, height, channels
        """
        return self._encode_model.input_shape[1:]

    @property
    def codebook_size(self):
        """
        Codebook size - how many vectors are used during the quantization operation
        """
        return self._quantizer.codebook_size

    @property
    def latent_space_size(self):
        """
        Tokens shape - token image width, height
        """
        return self._encode_model.output_shape[1:3]

    @property
    def downscale_multiplier(self) -> int:
        """
        How much is the image smaller in each dimension as tokens relative to the original size
        """
        img_size, token_size = self._encode_model.input_shape[1:3], self.latent_space_size
        return img_size[0]//token_size[0]

    def encode(self, image, training=False):
        """
        Convert the given image to tokens
        """
        return self._quantizer.get_tokens(self._quantizer.get_distances(self._encode_model(image, training)))

    def decode(self, tokens, training=False) -> tf.Tensor:
        """
        Convert the given tokens back to an image
        """
        return tf.clip_by_value(self._decode_model(self._quantizer.get_embedding(tokens), training=training), 0.0, 1.0)

    def get_config(self):
        return super().get_config() | {
            "img_size": self.img_size,
            "filters": self.filters,
            "residual_layer_multipliers": self.residual_layer_multipliers,
            "num_res_blocks": self.num_res_blocks,
            "discriminator_loss_weight": self.discriminator_loss_weight,
            "decoder_noise_dim": self.decoder_noise_dim,
        } | self._quantizer.get_config()

    @staticmethod
    def custom_objects():
        #custom loading function because automatic loading is broken for some reason
        def vqvae_load(*args2, **kwargs):
            kwargs.pop("layers")
            kwargs.pop("input_layers")
            kwargs.pop("output_layers")
            kwargs.pop("dtype")
            return VQVAEModel(*args2, **kwargs)
        return {"VQVAEModel":vqvae_load, "VectorQuantizer":VectorQuantizer, "scaled_mse_loss":scaled_mse_loss}

# pylint: enable=abstract-method




def main(args):
    #set the gpu to use, available threads, and the seed
    tf_init(args.use_gpu, args.threads, args.seed)

    class VQVAELog(TrainingLog):
        """
        Log how VQVAE performs
        """

        def run_test(self, data):
            """
            Log the VQVAE performance - show how it recreates images from the dataset and how it creates new ones
            """
            assert isinstance(self.model, VQVAEModel), "VQVAELog can only be used with VQVAEModel models"

            generated = self.model.generate_samples(10)
            reconstructed = self.model.predict(data, verbose=0)
            self.log.log_images("images", data).log_images("reconstruction", reconstructed).log_images("generated_images", generated)

    #load the model if required
    if not args.load_model:
        model = VQVAEModel.new(args)
    else:
        model = VQVAEModel.load(get_tokenizer_fname())

    #load the base dataset
    image_load = ImageLoading(args.dataset_location, args.img_size)

    #create train/dev/show datasets
    train_dataset, dev_dataset = image_load.create_train_dev_datasets(1000, args.batch_size)
    show_dataset = image_load.create_dataset(8)

    #start wandb logging
    WandbLog.wandb_init("image_outpainting_tokenizer", args)

    #create the logging class - log metrics every 25 batches, run validation / show images every 400
    model_log = VQVAELog(dev_dataset, show_dataset, 25, 400, learning_rate_decay=tf.optimizers.schedules.ExponentialDecay(1e-4, 1000000, 0.8))

    #train the model
    model.fit(train_dataset, epochs=args.epochs, callbacks=[
        WandbModelCheckpoint(get_tokenizer_fname(wandb.run.name), "reconstruction_loss", save_freq=20000),
        model_log]
    )


if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))
