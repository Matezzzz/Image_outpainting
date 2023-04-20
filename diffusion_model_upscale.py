import math
import argparse


import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbModelCheckpoint

from tokenizer import VQVAEModel, parser as tokenizer_args_parser
from utilities import get_tokenizer_fname, get_sharpening_fname, tf_init
from build_network import NetworkBuild as nb
from log_and_save import WandbLog, TrainingLog
from dataset import ImageLoading


parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--images_to_process", default=int(1e6), type=int, help="How many images should be processed in total. Used instead of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

parser.add_argument("--img_size", default=512, type=int, help="Input image size")
parser.add_argument("--residual_block_count", default=3, type=int, help="Number of residual blocks in sequence before a downscale")
parser.add_argument("--block_filter_counts", default=[32, 48, 64, 96, 160, 192, 256], nargs="+", type=int, help="Number of residual blocks at each resolution")
parser.add_argument("--noise_embed_dim", default=32, type=int, help="Sinusoidal embedding dimension")

parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")


parser.add_argument("--dataset_location", default="data", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], nargs="+", type=str, help="Individual places to use data from")

parser.add_argument("--load_model_run", default="", type=str, help="Name of the wandb run that created the model to load")




#I assume pure gaussian noise still contains 2% of the target signal, and the final image can still contain 5% noise
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95






#frequencies used for the sinusoidal embedding
EMBEDDING_MIN_FREQUENCY = 1.0
EMBEDDING_MAX_FREQUENCY = 1000.0
def sinusoidal_embedding(img_size, embedding_dims):
    """Return a function that computes a sinusoidal embedding"""
    # The frequencies of individual embedding components will be a geometric row
    frequencies = tf.experimental.numpy.geomspace(EMBEDDING_MIN_FREQUENCY, EMBEDDING_MAX_FREQUENCY, embedding_dims//2)
    # Legacy way of computing frequencies
    # frequencies = tf.exp(tf.linspace(tf.math.log(EMBEDDING_MIN_FREQUENCY), tf.math.log(EMBEDDING_MAX_FREQUENCY), embedding_dims // 2))
    angular_speeds = 2.0 * math.pi * frequencies

    def apply(x):
        #compute embeddings for one x
        embeddings = tf.concat([tf.sin(angular_speeds * x[:, tf.newaxis]), tf.cos(angular_speeds * x[:, tf.newaxis])], 1)
        #tile embeddings so they have the same size as the input image
        embeddings_whole = tf.tile(embeddings[:, tf.newaxis, tf.newaxis, :], [1, img_size, img_size, 1])
        return embeddings_whole
    return apply


#estimated mean & variance of images in the training set
IMAGE_MEAN = np.array([0.585, 0.668, 0.743], np.float32)
IMAGE_VARIANCE = np.array([0.03283161, 0.01849923, 0.01569985], np.float32)

#estimated mean & variance of produced edges (values that will be added to the blurred image in order to add detail)
EDGE_MEAN = np.array([0.00114195, -0.00257778,  0.00122407], np.float32)
EDGE_VARIANCE = np.array([0.00015055, 0.00013764, 0.00014862], np.float32)

# Code at https://keras.io/examples/generative/ddim/ was used as a starting point for this model
class DiffusionModel(tf.keras.Model):
    """Runs a diffusion model that is able to upscale images without losing detail"""
    def __init__(self, img_size, block_filter_counts, block_count, noise_embed_dim, learning_rate, weight_decay, train_batch_size, batch_size, *args, **kwargs):

        #create the diffusion model
        def create_diffusion_model(blurry_img, noisy_image, noise_variance):
            #add sinusoidal embeddings as additional filters to the first layer
            inp = nb.concat([blurry_img, noisy_image, noise_variance >> sinusoidal_embedding(img_size, noise_embed_dim)], -1)
            #pass everything through the u-net model, then do one final convolution to produce the results
            return inp >> nb.u_net(block_filter_counts, block_count, nb.layer_norm) >> nb.conv_2d(3, kernel_size=1)

        #create the model
        inp, out = nb.create_model_io((nb.inp([img_size, img_size, 3]), nb.inp([img_size, img_size, 3]), nb.inp([])), create_diffusion_model)
        super().__init__(inp, out, *args, **kwargs)

        #the tokenizer that will be used to train the model
        self._tokenizer = VQVAEModel.load(get_tokenizer_fname(), tokenizer_args_parser.parse_args([]))
        self._tokenizer.trainable = False

        self.img_size=img_size
        self.block_filter_counts = block_filter_counts
        self.block_count = block_count
        self.noise_embed_dim = noise_embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.batch_size = batch_size

        assert train_batch_size % batch_size == 0, "Train batch size must be divisible by batch size"
        #compute how many batches should be computed before updating weights
        steps_before_update = train_batch_size // batch_size

        #use the adam optimizer with weight decay & L1 error
        self.compile(
            optimizer=tf.optimizers.experimental.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
            loss=tf.losses.mean_absolute_error,
            steps_per_execution=steps_before_update
        )
        #self._ema_model = tf.keras.models.clone_model(self._model)

        #metrics - track noise loss and image loss
        self.noise_loss_tracker = tf.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = tf.metrics.Mean(name="image_loss")

    def _blur_images(self, images):
        """Take images, downscale them, pass them through the tokenizer, then upscale them back"""
        tok_img_s = self._tokenizer.img_size
        return tf.image.resize(self._tokenizer(tf.image.resize(images, [tok_img_s, tok_img_s])), [self.img_size, self.img_size], tf.image.ResizeMethod.BICUBIC)

    def _normalize_image(self, images):
        """Normalize image - move it so that the whole dataset has zero mean and unit variance"""
        return (images - IMAGE_MEAN) / tf.sqrt(IMAGE_VARIANCE)

    def _denormalize_image(self, images):
        """Undo the normalization operation - convert back into RGB between 0 and 1"""
        return tf.clip_by_value(images * tf.sqrt(IMAGE_VARIANCE) + IMAGE_MEAN, 0.0, 1.0)

    def _normalize_edges(self, images):
        """Normalize edges - move them so that all edges have zero mean and unit variance"""
        return (images - EDGE_MEAN) / tf.sqrt(EDGE_VARIANCE)

    def _denormalize_edges(self, images):
        """Undo the normalization operation on edges - convert back into RGB between 0 and 1"""
        return images * tf.sqrt(EDGE_VARIANCE) + EDGE_MEAN

    def _diffusion_schedule(self, diffusion_times) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute the cosine diffusion schedule - how much noise we expect to have in the image at provided diffusion times"""
        start_angle, end_angle = tf.acos(MAX_SIGNAL_RATE), tf.acos(MIN_SIGNAL_RATE)

        #the angle for each diffusion time
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # convert angles to noise_rates & signal_rates
        return tf.sin(diffusion_angles), tf.cos(diffusion_angles)

    def call(self, inputs, training=None, mask=None):
        """Call the model"""
        # Could use the _ema_model here if training was unstable
        return self(inputs, training=training)

    def denoise(self, blurry_image, noisy_image, noise_rates, signal_rates, training) -> tuple[tf.Tensor, tf.Tensor]:
        """Perform one denoising step on the given image"""
        # predict noise component and calculate the image component using it

        #predict the noise component on current inputs
        pred_noises = self([blurry_image, noisy_image, noise_rates**2], training=training)

        #estimate the image by subtracting the estimated noise component from it, and dividing by signal rate
        pred_image = (noisy_image - noise_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * pred_noises) / signal_rates[:, tf.newaxis, tf.newaxis, tf.newaxis]

        return pred_noises, pred_image

    def reverse_diffusion(self, blurry_image, initial_noise, diffusion_steps, target_mask = None, target_image = None):
        """Generate edges for the given blurry image, using many diffusion steps"""

        batch_size = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        #start with the noise
        next_noisy_image = initial_noise
        #the best image prediction so far
        pred_image = None
        for step in range(diffusion_steps):
            noisy_image = next_noisy_image

            # use the network to estimate the signal and noise in the current image
            diffusion_times = tf.ones([batch_size]) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_image = self.denoise(blurry_image, noisy_image, noise_rates, signal_rates, training=False)

            #if we have a target (i.e. some target values are already known), use them in place of the predicted image
            if target_mask is not None:
                pred_image = tf.where(target_mask, target_image, pred_image)

            # get the next noise and signal rates, and remove a slight amount of the predicted noise from the image so the rates fit in the next iteration
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_image = next_signal_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * pred_image + next_noise_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * pred_noises
        #return the image predicted during the last iteration
        return pred_image

    def improve_images(self, blurry_images, diffusion_steps, mask = None, target_image = None):
        """Provided with a blurry image, sharpen it"""
        #! try: add blurry images to initial noise?

        #create initial noise from which we will create the new edges using the model
        initial_noise = tf.random.normal(shape=tf.shape(blurry_images))
        #normalize blurry images
        blurry_images_norm = self.normalize_image(blurry_images)
        #the image we want to get, if known - e.g. if some parts of edges are finished already, and we are only working on part of the image
        target_edges = self.normalize_edges(target_image - blurry_images) if target_image is not None else None
        #use the model to predict the new edges
        edges = self.reverse_diffusion(blurry_images_norm, initial_noise, diffusion_steps, mask, target_edges)
        #denormalize the edges and add them to the image
        return tf.clip_by_value(blurry_images + self.denormalize_edges(edges), 0.0, 1.0)

    def improve_images_test(self, ideal_images, diffusion_steps):
        """Blur images, then improve them, and return both"""
        blurry_images = self.blur_images(ideal_images)
        return blurry_images, self.improve_images(blurry_images, diffusion_steps)

    def denoise_at_random_step(self, inputs, training):
        """Generate a random diffusion time step for all inputs, then try to make the model estimate the noise and image values"""
        ideal_image = inputs

        batch_size = tf.shape(ideal_image)[0]

        #blur the input image and normalize it
        blurry_image = self.blur_images(ideal_image)
        blurry_image_normed = self.normalize_image(blurry_image)

        #compute the edges and normalize them = the signal we want to predict
        edges = self.normalize_edges(ideal_image - blurry_image)
        #initial noise values
        noises = tf.random.normal(shape=(batch_size, self.img_size, self.img_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform([batch_size], minval=0.0, maxval=1.0)
        # get noise and signal rates according to the schendule
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        # mix the edge images (the signal) with noises accordingly
        noisy_images = signal_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * edges + noise_rates[:, tf.newaxis, tf.newaxis, tf.newaxis] * noises

        # train the network to separate noisy images into their noise and signal (edges) components
        pred_noises, pred_images = self.denoise(blurry_image_normed, noisy_images, noise_rates, signal_rates, training=training)

        #compute losses on both the computed noises and edge images
        noise_loss = self.compiled_loss(noises, pred_noises)
        image_loss = self.compiled_loss(edges, pred_images)

        # deprecated - ema (estimated moving average) network - could be used for slower, but more stable training
        # for weight, ema_weight in zip(self._model.weights, self._ema_model.weights):
        #     ema_weight.assign(0.999 * ema_weight + (1 - 0.999) * weight)
        return pred_noises, pred_images, noise_loss, image_loss


    def train_step(self, data):
        """Train on the provided dataset images"""
        ideal_image = data
        #train myself to be able to estimate the noise in the images at any diffusion step
        with tf.GradientTape() as tape:
            _, _, noise_loss, image_loss = self.denoise_at_random_step(ideal_image, training=True)
        #minimize the noise loss
        self.optimizer.minimize(noise_loss, self.trainable_variables, tape)

        # pylint: disable=not-callable
        #update noise and image loss trackers
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        #return current results
        return {"noise_loss":self.noise_loss_tracker.result(), "image_loss":self.image_loss_tracker.result()}
        # pylint: enable=not-callable

    def test_step(self, data):
        """Perform a test step - try denoising at a random step, measure the noise and image losses"""
        ideal_image = data
        _, _, noise_loss, image_loss = self.denoise_at_random_step(ideal_image, training=False)
        return {"test_noise_loss":noise_loss, "test_image_loss":image_loss}

    @staticmethod
    def load(path):
        """Load a DiffusionModel from the givne file"""

        # Custom loading function to fix tensorflow errors
        def load_diffusion_model(*args, **kwargs):
            kwargs.pop("layers")
            kwargs.pop("input_layers")
            kwargs.pop("output_layers")
            return DiffusionModel(*args, **kwargs, train_batch_size=2, batch_size=2)

        model = tf.keras.models.load_model(path, custom_objects={"DiffusionModel":load_diffusion_model, "VQVAEModel":VQVAEModel})

        #a.save_weights("sharpen_weights.h5")

        #diff_model = DiffusionModel.new(args)
        #diff_model.load_weights("sharpen_weights.h5")#, True)

        #diff_model.load_weights(path + "/variables", by_name=True)#, {"DiffusionModel":DiffusionModel})
        #diff_model.set_weights(model.get_weights())
        assert isinstance(model, DiffusionModel), "Loaded model must be a DiffusionModel"
        return model

    @staticmethod
    def new(args):
        """Create a new diffusion model from the given commandline arguments"""
        return DiffusionModel(args.img_size, args.block_filter_counts, args.residual_block_count, args.noise_embed_dim, args.learning_rate, args.weight_decay, args.train_batch_size, args.batch_size)

    def get_config(self):
        return super().get_config() | {
            "img_size":self.img_size,
            "block_filter_counts":self.block_filter_counts,
            "block_count":self.block_count,
            "noise_embed_dim":self.noise_embed_dim,
            "learning_rate":self.learning_rate,
            "weight_decay":self.weight_decay,
            "train_batch_size":self.train_batch_size,
            "batch_size": self.batch_size
        }





def main(args):
    #initialize tensorflow to use the given GPU, CPU thread count and seed
    tf_init(args.use_gpu, args.threads, args.seed)

    # create and compile the model
    model = DiffusionModel.new(args)

    #create a train, development and show datasets
    image_loading = ImageLoading(args.dataset_location, args.img_size, args.places, scale_down=1, shuffle_buffer=128)
    train_dataset, dev_dataset = image_loading.create_train_dev_datasets(1000, args.batch_size)
    sharpen_dataset = image_loading.create_dataset(4)

    # edge_dataset = train_dataset.map(lambda x:x-model.blur_images(x))

    # print ("Starting")
    # mean = image_loading.img_dataset_mean(edge_dataset, 100)
    # print (f"Mean = {mean}")
    # variance = image_loading.img_dataset_variance(edge_dataset, mean, 100)
    # print (f"Mean = {mean}, Variance = {variance}")

    #plot_image_variances(train_dataset)

    class DiffusionTrainingLog(TrainingLog):
        """Log training progress of the diffusion model"""

        def run_test(self, data):
            """Try sharpening some images from the dataset"""
            super().run_test(data)
            assert isinstance(self.model, DiffusionModel), "Model must be an instance of DiffusionModel"

            #sharpen the images
            blurry_images, sharpened = self.model.improve_images_test(data, diffusion_steps=100)

            #add them to the wandb log
            self.log.log_images("images", data).log_images("tokenizer_outputs", blurry_images).log_images("sharpened", sharpened)

        def run_validation(self):
            """Perform validation on the dev dataset, return the metrics"""
            return self.model.evaluate(self.dev_dataset, return_dict=True, verbose=0, steps=1000 // args.train_batch_size)

    #initialize wandb for logging
    WandbLog.wandb_init("image_outpainting_sharpen", args)

    # train the model
    model.fit(
        train_dataset.repeat(),
        epochs=1,
        callbacks=[
            WandbModelCheckpoint(filepath=get_sharpening_fname(wandb.run.name), monitor="val_image_loss", save_freq=20000//args.train_batch_size),
            DiffusionTrainingLog(dev_dataset, sharpen_dataset, log_frequency=25, test_frequency=5000//args.train_batch_size, train_batch_size=args.train_batch_size)
    ], steps_per_epoch = args.images_to_process // args.train_batch_size)



if __name__ == "__main__":
    given_arguments = parser.parse_args([] if "__file__" not in globals() else None)
    main(given_arguments)
