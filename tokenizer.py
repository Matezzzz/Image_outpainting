import os
GPU_TO_USE = int(open("gpu_to_use.txt").read().splitlines()[0]) # type: ignore
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if GPU_TO_USE == -1 else str(GPU_TO_USE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


from build_network import ResidualNetworkBuild as nb, Tensor
import numpy as np
import tensorflow as tf
from dataset import ImageLoading

import argparse

import log_and_save
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from discriminator import Discriminator

from utilities import get_tokenizer_fname

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
parser.add_argument("--filters", default=32, type=int, help="Base residual layer filters")
parser.add_argument("--residual_layer_multipliers", default=[1, 2, 3], nargs="+", type=int, help="How many residual layers to use and how to increase number of filters")
parser.add_argument("--num_res_blocks", default=2, type=int, help="Number of residual blocks in sequence before a downscale")
parser.add_argument("--embedding_dim", default=32, type=int, help="Embedding dimension for quantizer")
parser.add_argument("--codebook_size", default=128, type=int, help="Codebook size for quantizer")
parser.add_argument("--embedding_loss_weight", default=1.0, type=float, help="Scale the embedding loss")
parser.add_argument("--commitment_loss_weight", default=0.25, type=float, help="Scale for the commitment loss (penalize changing embeddings)")
parser.add_argument("--entropy_loss_weight", default=0.05, type=float, help="Scale for the entropy loss")
parser.add_argument("--discriminator_loss_weight", default=0.0, type=float, help="Scale for the discriminator loss")
# parser.add_argument("--entropy_loss_temperature", default=0.01, type=float, help="Entropy loss temperature")
# parser.add_argument("--data_dir", default="brno", type=str, help="Directory to read data from")
# parser.add_argument("--mask_fname", default="final_mask_brno.png", type=str, help="Segmentation mask for a directory")
# parser.add_argument("--load_model_dir", default=None, type=none_or_str, help="The model to load")


parser.add_argument("--dataset_location", default=".", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")
parser.add_argument("--load_model", default=True, type=none_or_str, help="The model to load")



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

    def call(self, inputs, training, **kwargs):
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
        self.add_metric(commitment_loss, "commitment_loss"); self.add_metric(embedding_loss, "embedding_loss"); self.add_metric(entropy_loss, "entropy_loss")
        if training: embeddings = inputs + tf.stop_gradient(embeddings - inputs)
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
            >> nb.residualDownscaleSequence(filters * np.array(residual_layer_multipliers), num_res_blocks, nb.batchNorm, nb.swish)\
            >> nb.groupNorm()\
            >> nb.swish\
            >> nb.conv2D(embedding_dim, kernel_size=1)
        return out.get()
    return encoder

def create_decoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim):
    def decoder(x):
        out = Tensor(x)\
            >> nb.conv2D(filters=embedding_dim)\
            >> nb.residualUpscaleSequence(filters * np.array(residual_layer_multipliers[::-1]), num_res_blocks, nb.batchNorm, nb.swish)\
            >> nb.groupNorm()\
            >> nb.swish\
            >> nb.conv2D(filters=3, activation="hard_sigmoid")
        return out.get()
    return decoder

data_variance = 0.0915055541062209
def scaled_mse_loss(y_true, y_pred): return tf.reduce_mean((y_true-y_pred)**2) / data_variance

class VQVAEModel(tf.keras.Model):
    def __init__(self, img_size, filters, residual_layer_multipliers, num_res_blocks, embedding_dim, codebook_size,
            embedding_loss_weight, commitment_loss_weight, entropy_loss_weight, discriminator_loss_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._quantizer = VectorQuantizer(codebook_size, embedding_dim, embedding_loss_weight, commitment_loss_weight, entropy_loss_weight)
        self.img_size = img_size
        self.filters = filters
        self.residual_layer_multipliers = residual_layer_multipliers
        self.num_res_blocks = num_res_blocks
        self.discriminator_loss_weight = discriminator_loss_weight

        inp_shape = [img_size, img_size, 3]
        self._encode_model = nb.create_model(nb.inpT(inp_shape), lambda x: x >> create_encoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim))
        self._decode_model = nb.create_model(nb.inpT(self._encode_model.output_shape[1:]), lambda x: x >> create_decoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim))
        self._model = nb.create_model(nb.inpT(inp_shape), lambda x: x >> self._encode_model >> self._quantizer >> self._decode_model)

        self.compile()

        #decay rate = 0.75 each epoch (65634 batches)
        self._model.compile(
            tf.optimizers.Adam(tf.optimizers.schedules.ExponentialDecay(1e-3, 65634, 0.75)), # type: ignore
            scaled_mse_loss,
            metrics=[tf.metrics.MeanSquaredError("reconstruction_loss")]
        )

        if self.discriminator_loss_weight != 0: self._discriminator = Discriminator(img_size, 2).get_model()


    def train_step(self, data):
        data_masked, data_true = data
        with tf.GradientTape(persistent=True) as tape:
            reconstruction = self._model(data_masked, training=True)
        
            batch_size = tf.shape(data_true)[0]

            if self.discriminator_loss_weight != 0:
                disc_inp = tf.concat([reconstruction, data_true], 0)
                disc_out = self._discriminator(disc_inp, training=True)
                
                disc_labels_fake = tf.zeros(batch_size)
                disc_labels_real = tf.ones(batch_size)
                disc_labels = tf.concat([disc_labels_fake, disc_labels_real], 0)
                
                disc_loss = self._discriminator.loss(disc_labels, disc_out) # type: ignore
                
                discriminator_loss = self.discriminator_loss_weight * self._discriminator.loss(disc_labels_real, disc_out[:batch_size]) # type: ignore
            else:
                #useless assigmnents to avoid undefined variable errors
                discriminator_loss, disc_loss, disc_inp, disc_labels, disc_out = 0, 0, 0, 0, 0
            
            reconstruction_loss = self._model.loss(data_true, reconstruction) # type: ignore
            vqvae_loss = sum(self._model.losses) # type: ignore
            
            loss = reconstruction_loss + vqvae_loss + discriminator_loss # type: ignore
        
        self._model.optimizer.minimize(
            loss, self._model.trainable_variables, tape
        )

        metrics = {'loss':loss, 'reconstruction_loss':reconstruction_loss, 'vqvae_discriminator_loss':discriminator_loss, 'learning_rate':self._model.optimizer.learning_rate} | self._model.compute_metrics(data_masked, data_true, reconstruction, tf.ones(batch_size))
        if self.discriminator_loss_weight != 0:
            metrics |= {'discriminator_loss':disc_loss} | self._discriminator.compute_metrics(disc_inp, disc_labels, disc_out, tf.ones(batch_size+batch_size))
            self._discriminator.optimizer.minimize(
                disc_loss, self._discriminator.trainable_variables, tape
            )
        return metrics

    def generate_samples(self, count):
        tokens = np.random.randint(0, self.codebook_size, [count, self._encode_model.output_shape[1], self._encode_model.output_shape[2]])
        embed_tokens = self._quantizer.get_embedding(tokens)
        return self._decode_model.predict(embed_tokens, verbose=0) #type: ignore

    def call(self, data):
        return self._model(data)

    @staticmethod
    def new(args):
        return VQVAEModel(args.img_size, args.filters, args.residual_layer_multipliers, args.num_res_blocks, args.embedding_dim, args.codebook_size,
                          args.embedding_loss_weight, args.commitment_loss_weight, args.entropy_loss_weight, args.discriminator_loss_weight)

    @staticmethod
    def load(fname, args) -> "VQVAEModel":
        vqvae_old = tf.keras.models.load_model(fname, {"VQVAEModel":VQVAEModel, "VectorQuantizer":VectorQuantizer, "scaled_mse_loss":scaled_mse_loss, "Discriminator":Discriminator}) #VQVAEModel.__new__(VQVAEModel)
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
    def codebook_size(self): return self._quantizer.codebook_size

    @property
    def latent_space_size(self): return self._encode_model.output_shape[1:3]
    
    @property
    def downscale_multiplier(self):
        a, b = self._encode_model.input_shape[1:3], self.latent_space_size
        return  a[0]//b[0], a[1]//b[1] 

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
        } | self._quantizer.get_config()
    



def main(args):
    if GPU_TO_USE == -1: tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)



    class ModelLog:
        def __init__(self, model, dataset):
            self._model = model
            self._dataset = iter(dataset)

        def log(self, batch, logs):
            wandb.log({"example_step": batch*args.batch_size}, commit=False)
            if batch % 400 == 0:
                generated = self._model.generate_samples(10)
                data, _ = next(self._dataset)
                reconstructed = self._model.predict(data, verbose=0)
                wandb.log({
                    "images": [wandb.Image(d) for d in data],
                    "reconstruction":[wandb.Image(d) for d in reconstructed],
                    "generated_images": [wandb.Image(d) for d in generated]
                }, commit=False)


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
    
    def repeat(x): return x, x
    
    image_load = ImageLoading(args.dataset_location, args.img_size, args.places, dataset_augmentation_base=repeat)

    train_dataset = image_load.create_dataset(args.batch_size)
    val_dataset = image_load.create_dataset(10)


    

    wandb_manager = log_and_save.WandbManager("image_outpainting_tokenizer")
    wandb_manager.start(args)
    model_log = ModelLog(model, val_dataset)

    model.fit(train_dataset, epochs=args.epochs, callbacks=[ #type: ignore
        WandbModelCheckpoint(get_tokenizer_fname(), "reconstruction_loss", save_freq=2500),
        tf.keras.callbacks.LambdaCallback(on_batch_end=model_log.log),
        WandbMetricsLogger(50)]
    )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)