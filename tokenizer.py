import argparse

# GPU_TO_USE = int(open("gpu_to_use.txt").read().splitlines()[0]) # type: ignore
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" if GPU_TO_USE == -1 else str(GPU_TO_USE)
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import tensorflow as tf
import wandb

from build_network import NetworkBuild as nb, Tensor
from dataset import ImageLoading
import log_and_save
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from discriminator import create_discriminator
from utilities import get_tokenizer_fname, tf_init

def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--filters", default=32, type=int, help="Base residual layer filters")
parser.add_argument("--residual_layer_multipliers", default=[1, 2, 3], nargs="+", type=int, help="How many residual layers to use and how to increase number of filters")
parser.add_argument("--num_res_blocks", default=2, type=int, help="Number of residual blocks in sequence before a downscale")
parser.add_argument("--embedding_dim", default=32, type=int, help="Embedding dimension for quantizer")
parser.add_argument("--codebook_size", default=128, type=int, help="Codebook size for quantizer")
parser.add_argument("--embedding_loss_weight", default=0.05, type=float, help="Scale the embedding loss")
parser.add_argument("--commitment_loss_weight", default=0.02, type=float, help="Scale for the commitment loss (penalize changing embeddings)")
parser.add_argument("--entropy_loss_weight", default=0.01, type=float, help="Scale for the entropy loss")
parser.add_argument("--discriminator_loss_weight", default=0.0, type=float, help="Scale for the discriminator loss")
parser.add_argument("--decoder_noise_dim", default=10, type=int, help="How many dimensions of loss to add before decoding")
# parser.add_argument("--entropy_loss_temperature", default=0.01, type=float, help="Entropy loss temperature")
# parser.add_argument("--data_dir", default="brno", type=str, help="Directory to read data from")
# parser.add_argument("--mask_fname", default="final_mask_brno.png", type=str, help="Segmentation mask for a directory")
# parser.add_argument("--load_model_dir", default=None, type=none_or_str, help="The model to load")


parser.add_argument("--dataset_location", default=".", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")
parser.add_argument("--load_model", default=False, type=bool, help="Whether to load model or not")



class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, codebook_size, embedding_dim, embedding_loss_weight, commitment_loss_weight, entropy_loss_weight, **kwargs):
        super().__init__(**kwargs)
        #these need to be easily accessed in the training loop
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embedding_loss_weight = embedding_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook = self.add_weight("codebook", [codebook_size, embedding_dim], tf.float32, tf.keras.initializers.RandomUniform())

    @staticmethod
    def new(args):
        return VectorQuantizer(
            args.codebook_size,
            args.embedding_dim,
            args.embedding_loss_weight,
            args.commitment_loss_weight,
            args.entropy_loss_weight
        )

    def get_embedding(self, tokens):
        #batch dims = [batch, width, height]
        return tf.gather(self.codebook, tokens, axis=0)

    def get_distances(self, values):
        #Dims -> [batch, w, h, 1, embed] - [codebook_size, embed] ->  [batch, w, h, codebook_size, embed];
        #reduce_sum -> [batch, w, h, codebook_size]
        return tf.reduce_sum(tf.square(values[:, :, :, tf.newaxis, :] - self.codebook), -1)

    #values = [batch, w, h, embed]
    def get_tokens(self, distances):
        return tf.argmin(distances, -1, tf.int32)



    def call(self, inputs, *args, training=None, **kwargs):
        distances = self.get_distances(inputs)
        tokens = self.get_tokens(distances)
        embeddings = self.get_embedding(tokens)

        embedding_loss = self.embedding_loss_weight * tf.reduce_mean((tf.stop_gradient(embeddings) - inputs)**2)
        commitment_loss = self.commitment_loss_weight * tf.reduce_mean((embeddings - tf.stop_gradient(inputs))**2)
        #move all embeddings to their closest vector

        #distances = [batch, w, h, codebook_size]

        #classes

        #entropy_loss = self.entropy_loss_weight * tf.reduce_mean(tf.reduce_min(tf.reshape(distances, [-1, self.codebook_size]), 0)) #self.compute_entropy_loss(distances)
        flat_distances = tf.reshape(distances, [-1, self.codebook_size])
        # [batch*w*h]
        closest_classes = tf.argmin(flat_distances, 1, tf.int32)
        # [codebook_size]
        active_classes = tf.scatter_nd(closest_classes[:, tf.newaxis], tf.ones_like(closest_classes), [self.codebook_size])

        entropy_loss = self.entropy_loss_weight * tf.reduce_mean(tf.where(active_classes == 0, tf.reduce_min(flat_distances, 0), 0.0), 0)

        self.add_loss(commitment_loss + embedding_loss + entropy_loss)
        self.add_metric(commitment_loss, "commitment_loss")
        self.add_metric(embedding_loss, "embedding_loss")
        self.add_metric(entropy_loss, "entropy_loss")
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
    def encoder(x):
        out = Tensor(x)\
            >> nb.residual_downscale_sequence(filters * np.array(residual_layer_multipliers), num_res_blocks, nb.batch_norm, nb.swish)\
            >> nb.group_norm()\
            >> nb.swish\
            >> nb.conv_2d(embedding_dim, kernel_size=1)
        return out.get()
    return encoder

def create_decoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim, decoder_noise_dim):
    def decoder(x):
        def add_noise_dims(x):
            #x = [batch, tok_x, tok_y, channels]
            if decoder_noise_dim > 0:
                shape = tf.shape(x)
                return tf.concat([x, tf.random.normal([shape[0], shape[1], shape[2], decoder_noise_dim])], -1)
            return x
        out = Tensor(x)\
            >> add_noise_dims\
            >> nb.conv_2d(filters=embedding_dim)\
            >> nb.residual_upscale_sequence(filters * np.array(residual_layer_multipliers[::-1]), num_res_blocks, nb.batch_norm, nb.swish)\
            >> nb.group_norm()\
            >> nb.swish\
            >> nb.conv_2d(filters=3, activation="hard_sigmoid")
        return out.get()
    return decoder

DATA_VARIANCE = 0.0915055541062209
def scaled_mse_loss(y_true, y_pred):
    return tf.reduce_mean((y_true-y_pred)**2) / DATA_VARIANCE




class VQVAEModel(tf.keras.Model):
    def __init__(self, img_size, filters, residual_layer_multipliers, num_res_blocks, embedding_dim, codebook_size, decoder_noise_dim,
            embedding_loss_weight, commitment_loss_weight, entropy_loss_weight, discriminator_loss_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._quantizer = VectorQuantizer(codebook_size, embedding_dim, embedding_loss_weight, commitment_loss_weight, entropy_loss_weight)
        self.img_size = img_size
        self.filters = filters
        self.residual_layer_multipliers = residual_layer_multipliers
        self.num_res_blocks = num_res_blocks
        self.discriminator_loss_weight = discriminator_loss_weight
        self.decoder_noise_dim = decoder_noise_dim

        inp_shape = [img_size, img_size, 3]
        self._encode_model = nb.create_model(nb.inp(inp_shape),
                                             lambda x: x >> create_encoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim))

        self._decode_model = nb.create_model(nb.inp(self._encode_model.output_shape[1:]),
                                             lambda x: x >> create_decoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim, decoder_noise_dim))

        self._model = nb.create_model(nb.inp(inp_shape), lambda x: x >> self._encode_model >> self._quantizer >> self._decode_model)

        self.compile()

        #decay rate = 0.75 each epoch (65634 batches)
        self._model.compile(
            tf.optimizers.Adam(tf.optimizers.schedules.ExponentialDecay(1e-3, 65634, 0.75)), # type: ignore
            scaled_mse_loss
        )

        if self.discriminator_loss_weight != 0:
            self._discriminator = create_discriminator(img_size, 2)



    def train_step(self, data):
        metrics = {}

        with tf.GradientTape() as tape:
            reconstruction = self._model(data, training=True)

            batch_size = tf.shape(data)[0]
            if self.discriminator_loss_weight != 0:
                disc_out = self._discriminator(reconstruction, training=True)
                discriminator_loss = self.discriminator_loss_weight * self._discriminator.compiled_loss(tf.ones(batch_size), disc_out)
                metrics |= {'discriminator_loss':discriminator_loss}
            else:
                discriminator_loss = 0

            reconstruction_loss = self._model.compiled_loss(data, reconstruction)
            vqvae_loss = sum(self._model.losses)

            loss = reconstruction_loss + discriminator_loss + vqvae_loss
            metrics |= {'main_loss':loss, 'reconstruction_loss': reconstruction_loss, 'vqvae_loss':vqvae_loss, 'learning_rate':self._model.optimizer.learning_rate}
            self._model.compiled_metrics.update_state(data, reconstruction)
            metrics |= {m.name:m.result() for m in self._model.metrics}

        self._model.optimizer.minimize(loss, self._model.trainable_variables, tape)

        if self.discriminator_loss_weight != 0:
            with tf.GradientTape() as tape:
                disc_labels = tf.concat([tf.zeros(batch_size), tf.ones(batch_size)], 0)
                disc_input = tf.concat([reconstruction, data], 0)

                disc_predictions = self._discriminator(disc_input, training=True)

                disc_loss = self._discriminator.compiled_loss(disc_labels, disc_predictions)
            self._discriminator.optimizer.minimize(disc_loss, self._discriminator.trainable_variables, tape)
            self._discriminator.compiled_metrics.update_state(disc_labels, disc_predictions)
            metrics |= {'D_discriminator_loss': disc_loss} | {("D_"+m.name):m.result() for m in self._discriminator.metrics}
        return metrics

    def generate_samples(self, count):
        tokens = np.random.randint(0, self.codebook_size, [count, self._encode_model.output_shape[1], self._encode_model.output_shape[2]])
        embed_tokens = self._quantizer.get_embedding(tokens)
        return self._decode_model.predict(embed_tokens, verbose=0) #type: ignore

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training)

    @staticmethod
    def new(args):
        return VQVAEModel(args.img_size, args.filters, args.residual_layer_multipliers, args.num_res_blocks, args.embedding_dim, args.codebook_size, args.decoder_noise_dim,
                          args.embedding_loss_weight, args.commitment_loss_weight, args.entropy_loss_weight, args.discriminator_loss_weight)

    @staticmethod
    def load(fname, args : argparse.Namespace) -> "VQVAEModel":
        def vqvae_load_old(*args2, **kwargs):
            return VQVAEModel(*args2, **kwargs, decoder_noise_dim=args.decoder_noise_dim)

        vqvae_old = tf.keras.models.load_model(fname, {"VQVAEModel":vqvae_load_old, "VectorQuantizer":VectorQuantizer, "scaled_mse_loss":scaled_mse_loss}) #VQVAEModel.__new__(VQVAEModel)
        vqvae = VQVAEModel.new(args)
        vqvae.set_weights(vqvae_old.get_weights()) # type: ignore
        #vqvae = VQVAEModel(vqvae_old.img_size, vqvae_old.filters, vqvae_old.residual_layer_multipliers, vqvae_old.num_res_blocks, vqvae_old.embedding_dim, vqvae_old.codebook_size, )
        #super(VQVAEModel, vqvae).__init__() # type: ignore
        #vqvae.compile()
        #vqvae._model = #tf.keras.models.load_model(fname, {"VQVAEModel":VQVAEModel, "VectorQuantizer":VectorQuantizer, "scaled_mse_loss":scaled_mse_loss, "Discriminator":Discriminator})
        #vqvae._encode_model = vqvae._model.get_layer(index=1) #type: ignore
        #vqvae._quantizer = vqvae._model.get_layer("vector_quantizer") #type: ignore
        #vqvae._decode_model = vqvae._model.get_layer(index=2) #type: ignore
        return vqvae # type: ignore

    @property
    def input_shape(self):
        return self._encode_model.input_shape[1:]

    @property
    def codebook_size(self):
        return self._quantizer.codebook_size

    @property
    def latent_space_size(self):
        return self._encode_model.output_shape[1:3]

    @property
    def downscale_multiplier(self):
        img_size, token_size = self._encode_model.input_shape[1:3], self.latent_space_size
        return img_size[0]//token_size[0], img_size[1]//token_size[1]

    def encode(self, image, training=False):
        return self._quantizer.get_tokens(self._quantizer.get_distances(self._encode_model(image, training)))

    def decode(self, tokens, training=False) -> tf.Tensor:
        return self._decode_model(self._quantizer.get_embedding(tokens), training=training) #type: ignore

    def get_config(self):
        return super().get_config() | {
            "img_size": self.img_size,
            "filters": self.filters,
            "residual_layer_multipliers": self.residual_layer_multipliers,
            "num_res_blocks": self.num_res_blocks,
            "discriminator_loss_weight": self.discriminator_loss_weight,
            "decoder_noise_dim": self.decoder_noise_dim,
        } | self._quantizer.get_config()




def main(args):
    tf_init(args.use_gpu, args.threads, args.seed)



    class ModelLog(tf.keras.callbacks.Callback):
        def __init__(self, model, dataset):
            super().__init__()
            self._model = model
            self._dataset = iter(dataset)

        def on_batch_end(self, batch, logs=None):
            wandb.log({"example_step": batch*args.batch_size}, commit=False)
            if batch % 400 == 0:
                generated = self._model.generate_samples(10)
                data = next(self._dataset)
                reconstructed = self._model.predict(data, verbose=0)
                wandb.log({
                    "images": [wandb.Image(d) for d in data],
                    "reconstruction":[wandb.Image(d) for d in reconstructed],
                    "generated_images": [wandb.Image(d) for d in generated]
                }, commit=False)
            return super().on_batch_end(batch, logs)


    # image_size = np.array([1600, 1200]) // 4

    # mask = np.asarray(Image.open(args.mask_fname).resize(tuple(image_size)))
    # data_boxes = []
    # while len(data_boxes) < 10:
    #     x, y = np.random.randint(0, image_size - 1 - args.img_size, 2)
    #     if np.sum(mask[x:x+args.img_size, y:y+args.img_size]) == 0:
    #         data_boxes.append([x, y])


    # def create_dataset(batch_size):
    #     d = tf.keras.utils.image_dataset_from_directory(args.data_dir, labels=None, batch_size=None, image_size=image_size.T) # type: ignore
    #     def select_areas(img): return tf.data.Dataset.from_tensor_slices([img[a:a+args.img_size,b:b+args.img_size] / 255.0 for a, b in data_boxes])
    #     def filter(img): return tf.reduce_mean(tf.reduce_mean(img, -1)) > 0.2
    #     def get_data(img): return img, img

    #     d = d.flat_map(select_areas).filter(filter).map(get_data) #type: ignore
    #     return d.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if not args.load_model:
        model = VQVAEModel.new(args)
    else:
        model = VQVAEModel.load(get_tokenizer_fname(), args)

    image_load = ImageLoading(args.dataset_location, args.img_size, args.places)#, dataset_augmentation_base=repeat)

    train_dataset = image_load.create_dataset(args.batch_size)
    val_dataset = image_load.create_dataset(10)

    wandb_manager = log_and_save.WandbManager("image_outpainting_tokenizer")
    wandb_manager.start(args)
    model_log = ModelLog(model, val_dataset)

    model.fit(train_dataset, epochs=args.epochs, callbacks=[
        WandbModelCheckpoint(get_tokenizer_fname(wandb.run.name), "reconstruction_loss", save_freq=2500), #type: ignore
        model_log,
        WandbMetricsLogger(50)]
    )


if __name__ == "__main__":
    _cmdline_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(_cmdline_args)
