import math
import argparse


import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from tokenizer import VQVAEModel, parser as tokenizer_args_parser
from utilities import get_tokenizer_fname, get_sharpening_fname
from build_network import ResidualNetworkBuild as nb
from log_and_save import WandbManager
from dataset import ImageLoading


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--residual_block_count", default=2, type=int, help="Number of residual blocks in sequence before a downscale")
parser.add_argument("--block_filter_counts", default=[32, 64, 96, 128], nargs="+", type=int, help="Number of residual blocks at each resolution")
parser.add_argument("--noise_embed_dim", default=32, type=int, help="Sinusoidal embedding dimension")

parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")


parser.add_argument("--dataset_location", default=".", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")

parser.add_argument("--load_model_run", default="", type=str, help="Name of the wandb run that created the model to load")




# sampling
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95

EMBEDDING_MIN_FREQUENCY = 1.0
EMBEDDING_MAX_FREQUENCY = 1000.0




def sinusoidal_embedding(img_size, embedding_dims):
    def apply(x):
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(EMBEDDING_MIN_FREQUENCY),
                tf.math.log(EMBEDDING_MAX_FREQUENCY),
                embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat([tf.sin(angular_speeds * x[:, tf.newaxis]), tf.cos(angular_speeds * x[:, tf.newaxis])], 1)

        embeddings_whole = tf.tile(embeddings[:, tf.newaxis, tf.newaxis, :], [1, img_size, img_size, 1]) #type: ignore

        return embeddings_whole
    return apply




# Code greatly inspired by https://keras.io/examples/generative/ddim/
class DiffusionModel(tf.keras.Model):
    def __init__(self, img_size, block_filter_counts, block_count, noise_embed_dim, learning_rate, weight_decay):
        super().__init__()

        self.img_size=img_size
        self.block_filter_counts = block_filter_counts
        self.block_count = block_count
        self.noise_embed_dim = noise_embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        def create_diffusion_model(noisy_img, noise_variance):
            inp = nb.concat([noisy_img, noise_variance >> sinusoidal_embedding(img_size, noise_embed_dim)], -1) # type: ignore
            return inp >> nb.u_net(block_filter_counts, block_count) >> nb.conv2D(3, kernel_size=1)
            #return\
            #    >> nb.residualSequence(block_count, filter_count)\
            #    >> nb.conv2D(3, kernel_size=1)

        self.normalizer = tf.keras.layers.Normalization()
        self._model = nb.create_model((nb.inpT([img_size, img_size, 3]), nb.inpT([])), create_diffusion_model)
        self._model.compile(
            optimizer=tf.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
            loss=tf.losses.mean_absolute_error
        )
        self._ema_model = tf.keras.models.clone_model(self._model)
        self.noise_loss_tracker = tf.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = tf.metrics.Mean(name="image_loss")
        self.compile()

    def normalize(self, images, training):
        return self.normalizer(images, training=training)

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times) -> tuple[tf.Tensor, tf.Tensor]:
        # diffusion times -> angles
        #float
        start_angle = tf.acos(MAX_SIGNAL_RATE)
        #float
        end_angle = tf.acos(MIN_SIGNAL_RATE)
        #[batch]
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates # type: ignore

    def denoise(self, noisy_images, noise_rates, signal_rates, training) -> tuple[tf.Tensor, tf.Tensor]:
        model = self._model if training else self._ema_model
        # predict noise component and calculate the image component using it
        pred_noises = model([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * pred_noises) / signal_rates[:, tf.newaxis, tf.newaxis, tf.newaxis]

        return pred_noises, pred_images # type: ignore

    def reverse_diffusion(self, initial_noise, diffusion_steps, initial_noise_level=1.0):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        pred_images = None
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = (tf.ones([num_images]) - step * step_size) * initial_noise_level
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size * initial_noise_level
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * pred_images + next_noise_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * pred_noises # type: ignore
            # this new noisy image will be used in the next step
        return pred_images
    
    def improve_images(self, images, initial_noise_level, diffusion_steps):
        initial_noise = tf.random.normal(shape=tf.shape(images))
        inp = self.normalize(images, False)
        fixed_images = self.reverse_diffusion(initial_noise * initial_noise_level + (1 - initial_noise_level) * inp, diffusion_steps, initial_noise_level)
        return self.denormalize(fixed_images)

    def generate(self, num_images, diffusion_steps) -> tf.Tensor:
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, self.img_size, self.img_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images # type: ignore

    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        # normalize images to have standard deviation of 1, like the noises
        #[batch, w, h, 3]
        images = self.normalize(inputs, training=training) # type: ignore
        #[batch, w, h, 3]
        noises = tf.random.normal(shape=(batch_size, self.img_size, self.img_size, 3))

        # sample uniform random diffusion times
        # [batch]
        diffusion_times = tf.random.uniform([batch_size], minval=0.0, maxval=1.0) # type: ignore
        #[batch], [batch]
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the images with noises accordingly
        noisy_images = signal_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + noise_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * noises # type: ignore

        # train the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=training)

        noise_loss = self._model.compiled_loss(noises, pred_noises)  # type: ignore
        image_loss = self._model.compiled_loss(images, pred_images)  # type: ignore
        
        for weight, ema_weight in zip(self._model.weights, self._ema_model.weights):
            ema_weight.assign(0.999 * ema_weight + (1 - 0.999) * weight)

        return pred_noises, pred_images, noise_loss, image_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            _, _, noise_loss, image_loss = self(data, training=True) #type: ignore

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        self._model.optimizer.minimize(noise_loss, self._model.trainable_variables, tape)
        return {"noise_loss":self.noise_loss_tracker.result(), "image_loss":self.image_loss_tracker.result()}

    def test_step(self, data):
        _, _, noise_loss, image_loss = self(data, training=False) #type: ignore
        return {"test_noise_loss":noise_loss, "test_image_loss":image_loss}

    @staticmethod
    def new(args):
        return DiffusionModel(args.img_size, args.block_filter_counts, args.residual_block_count, args.noise_embed_dim, args.learning_rate, args.weight_decay)

    def get_config(self):
        return super().get_config() | {
            "img_size":self.img_size,
            "block_filter_counts":self.block_filter_counts,
            "block_count":self.block_count,
            "noise_embed_dim":self.noise_embed_dim,
            "learning_rate":self.learning_rate,
            "weight_decay":self.weight_decay
        }





def main(args):
    tf.config.set_visible_devices((tf.config.get_visible_devices("GPU")[args.use_gpu] if args.use_gpu != -1 else []), "GPU")

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # create and compile the model
    model = DiffusionModel.new(args)


    image_loading = ImageLoading(args.dataset_location, args.img_size, args.places)

    train_dataset, dev_dataset = image_loading.create_train_dev_datasets(1000, args.batch_size)
    sharpen_dataset = image_loading.create_dataset(10)


    class ModelLog(tf.keras.callbacks.Callback):
        def __init__(self, model : DiffusionModel, dev_dataset, sharpen_dataset):
            super().__init__()
            self._model = model
            self._dev_dataset = dev_dataset
            self._sharpen_dataset = iter(sharpen_dataset.repeat(None))
            self._tokenizer = VQVAEModel.load(get_tokenizer_fname(), tokenizer_args_parser.parse_args([]))

        def on_batch_end(self, batch, logs=None):
            if batch % 400 == 0:
                inputs = next(self._sharpen_dataset)
                outputs = self._tokenizer(inputs)
                sharpened_10 = self._model.improve_images(outputs, initial_noise_level=0.1, diffusion_steps=20)
                sharpened_25 = self._model.improve_images(outputs, initial_noise_level=0.25,diffusion_steps=20)
                sharpened_50 = self._model.improve_images(outputs, initial_noise_level=0.5, diffusion_steps=20)
                generated = self._model.generate(num_images=8, diffusion_steps=20)

                wandb.log({
                    "images":[wandb.Image(i) for i in inputs],
                    "tokenizer_outputs":[wandb.Image(i) for i in outputs], # type: ignore
                    "sharpened_10":[wandb.Image(i) for i in sharpened_10],
                    "sharpened_25":[wandb.Image(i) for i in sharpened_25],
                    "sharpened_50":[wandb.Image(i) for i in sharpened_50],
                    "generated":[wandb.Image(i) for i in generated]}, commit=False)
                if logs is None:
                    logs = {}
                val_logs = self._model.evaluate(self._dev_dataset, return_dict=True, verbose=0) # type: ignore
                return logs | {("val_"+name):val for name, val in val_logs.items()} # type: ignore
            return super().on_batch_end(batch, logs)

    WandbManager("image_outpainting_sharpen").start(args)
    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset.take(200))


    # run training and plot generated images periodically
    model.fit(
        train_dataset,
        epochs=args.epochs,
        callbacks=[
            WandbModelCheckpoint(filepath=get_sharpening_fname(wandb.run.name), monitor="val_image_loss"), #type: ignore
            ModelLog(model, dev_dataset, sharpen_dataset),
            WandbMetricsLogger(25)
    ])

if __name__ == "__main__":
    given_arguments = parser.parse_args([] if "__file__" not in globals() else None)
    main(given_arguments)
