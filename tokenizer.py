from build_network import ResidualNetworkBuild as nb, TensorflowOp, NetworkOp
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import random

import argparse

import log_and_save
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--activation", default="swish", type=str, help="Activation func for convolutions")
parser.add_argument("--filters", default=32, type=int, help="Base residual layer filters")
parser.add_argument("--residual_layer_multipliers", default=[1, 2, 3], type=list, help="How many residual layers to use and how to increase number of filters")
parser.add_argument("--num_res_blocks", default=2, type=int, help="Number of residual blocks in sequence before a downscale")
parser.add_argument("--embedding_dim", default=32, type=int, help="Embedding dimension for quantizer")
parser.add_argument("--codebook_size", default=128, type=int, help="Codebook size for quantizer")
parser.add_argument("--commitment_loss_weight", default=0.25, type=float, help="Scale for the commitment loss (penalize changing embeddings)")
parser.add_argument("--entropy_loss_weight", default=0.1, type=float, help="Scale for the entropy loss")
parser.add_argument("--entropy_loss_temperature", default=0.01, type=float, help="Entropy loss temperature")








class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, codebook_size, embedding_dim, commitment_loss_weight, entropy_loss_weight, entropy_loss_temperature, **kwargs):
        super().__init__(**kwargs)
        self._args = args
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.codebook = self.add_weight("codebook", [args.codebook_size, args.embedding_dim], tf.float32, tf.keras.initializers.RandomUniform(-5, 5))
        self._mse = tf.keras.losses.MeanSquaredError()
    
    def get_embedding(self, tokens):
        #batch dims = [batch, width, height]
        return tf.gather(self.codebook, tokens, axis=0)
    
    def get_distances(self, values):
        #Dims -> [batch, w, h, 1, embed] - [codebook_size, embed] ->  [batch, w, h, codebook_size, embed];
        #reduce_sum -> [batch, w, h, codebook_size]
        return tf.reduce_sum(tf.square(values[:, :, :, tf.newaxis, :] - self.codebook), -1)
    
    #values = [batch, w, h, embed]
    def get_tokens(self, distances):
        return tf.argmin(distances, -1)
    
    def call(self, inputs, training, **kwargs):
        distances = self.get_distances(inputs)
        tokens = self.get_tokens(distances)
        embeddings = self.get_embedding(tokens)
        self.add_loss(self.entropy_loss_weight * self.compute_entropy_loss(distances))
        self.add_loss(self._mse(tf.stop_gradient(inputs), embeddings) * self.commitment_loss_weight)
        self.add_loss(self._mse(tf.stop_gradient(embeddings), inputs))
        if training: embeddings = inputs + tf.stop_gradient(embeddings - inputs)
        return embeddings
    
        
    def compute_entropy_loss(self, distances):
        #shape [all_points, codebook_size]
        affinity = tf.reshape(-distances, [-1, tf.shape(distances)[-1]]) / self._args.entropy_loss_temperature
        probs = tf.nn.softmax(affinity, -1)
        log_probs = tf.nn.log_softmax(affinity+1e-5, -1)
        avg_probs = tf.reduce_mean(probs, 0)
        avg_entropy = -tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-5))
        sample_entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, -1)) # type: ignore
        loss = sample_entropy - avg_entropy
        return loss
    

    def get_config(self):
        config = super().get_config()
        return config.update({
            "codebook_size":self.codebook_size,
            "embedding_dim":self.embedding_dim,
            "commitment_loss_weight":self.commitment_loss_weight,
            "entropy_loss_weight":self.entropy_loss_weight,
            "entropy_loss_temperature":self.entropy_loss_temperature
        })
    
    
        


def create_encoder(args):
    return NetworkOp(lambda x: nb.conv2DDown(filters=10)(x)\
        >> nb.residualDownscaleSequence(args.filters * np.array(args.residual_layer_multipliers), args.num_res_blocks, nb.groupnorm, nb.swish)\
        >> nb.groupnorm()\
        >> nb.swish()\
        >> nb.conv2D(args.embedding_dim, kernel_size=1)
    )

def create_decoder(args):
    return NetworkOp(lambda x:nb.conv2D(filters=args.embedding_dim)(x)\
        >> nb.residualUpscaleSequence(args.filters * np.array(args.residual_layer_multipliers[::-1]), args.num_res_blocks, nb.groupnorm, nb.swish)\
        >> nb.groupnorm()\
        >> nb.swish()\
        >> nb.conv2DUp(filters=6, activation="relu")\
        >> nb.conv2D(filters=3, activation="hard_sigmoid")
    )



class VQVAEModel(tf.keras.Model):
    def __init__(self, args : argparse.Namespace):
        super().__init__()
        self._quantizer = VectorQuantizer(
            args.codebook_size,
            args.embedding_dim,
            args.commitment_loss_weight,
            args.entropy_loss_weight,
            args.entropy_loss_temperature
        )
        inp_shape = [args.img_size, args.img_size, 3]
        self._encode_model = nb.inp(inp_shape) >> create_encoder(args) >> nb.model()
        self._decode_model = nb.inp(self._encode_model.output_shape[1:]) >> create_decoder(args) >> nb.model()
        self._mse = tf.losses.MeanSquaredError()
        self._args = args
        self._optimizer = tf.keras.optimizers.Adam()
        self.compile()
        
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            #encoded, tokens, tokens_embed, decoded, entropy_loss = self.run_model(data, True)
            
            reconstruction_loss = self._mse(data, self(data, True))
            #e_latent_loss = self._mse(tf.stop_gradient(tokens_embed), encoded) * self._args.commitment_loss_weight
            #q_latent_loss = self._mse(tf.stop_gradient(encoded), tokens_embed)

            e_latent_loss, q_latent_loss, entropy_loss = self._quantizer.losses
            loss = reconstruction_loss + e_latent_loss + q_latent_loss + entropy_loss
            
        self._optimizer.minimize(
            loss,
            (self._encode_model.trainable_variables, self._decode_model.trainable_variables, self._quantizer.trainable_variables),
            tape=tape
        )
        losses = {"reconstruction_loss": reconstruction_loss, "e_latent_loss":e_latent_loss, "q_latent_loss":q_latent_loss, "entropy_loss":entropy_loss}
        return losses
        
    def generate_samples(self, count):
        tokens = np.random.randint(0, self._args.codebook_size, [count, self._encode_model.output_shape[1], self._encode_model.output_shape[2]])
        embed_tokens = self._quantizer.get_embedding(tokens)
        return self._decode_model.predict(embed_tokens)
    
    def call(self, data, training):
        return self._decode_model(self._quantizer(self._encode_model(data, training), training), training)
        # encoded = self._encode_model(data)
        # tokens, entropy_loss = self._quantizer.get_tokens(encoded, train)
        # tokens_embed = self._quantizer.get_embedding(tokens)
        # if train: tokens_embed = tokens_embed + tf.stop_gradient(encoded - tokens_embed)
        # decoded = self._decode_model(tokens_embed)
        # return (encoded, tokens, tokens_embed, decoded, entropy_loss)
    
    # def call(self, data, training, mask):
    #     return self(data, training)
    def get_config(self):
        config = super().get_config()
        return config | {
            
        }
            
def main(args):
    #tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    

    
    class ModelLog:
        def __init__(self, model, dataset):
            self._model = model
            self._dataset = iter(dataset)

        def log(self, batch, logs):
            if batch % 400 == 0:
                generated = self._model.generate_samples(10)
                data = next(self._dataset)
                reconstructed = self._model.predict(data)
                wandb.log({"images": [wandb.Image(d) for d in data], "reconstruction":[wandb.Image(d) for d in reconstructed], "generated_images": [wandb.Image(d) for d in generated]})
    
    
    
    image_size = np.array([1600, 1200]) // 4
    
    mask = np.asarray(Image.open("masks/final_mask_brno.png").resize(tuple(image_size)))
    data_boxes = []
    while len(data_boxes) < 10:
        x, y = np.random.randint(0, image_size - 1 - args.img_size, 2)
        if np.sum(mask[x:x+args.img_size, y:y+args.img_size]) == 0:
            data_boxes.append([x, y])
        
    
    image_fnames = glob.glob("brno/*/*.jpg")
    random.shuffle(image_fnames)
    def create_dataset(batch_size):
        def load_images():
            for i in image_fnames:
                img = np.asarray(Image.open(i).resize(tuple(image_size))) / 255.0
                for a, b in data_boxes:
                    x = img[a:a+args.img_size,b:b+args.img_size]
                    yield x.astype(np.float32)
        
        #d = tf.keras.utils.image_dataset_from_directory("brno", labels=None, batch_size=None, image_size=image_size.T) # type: ignore
        #def select_areas(img):
        #    return tf.data.Dataset.from_tensor_slices([img / 255.0 for a, b in data_boxes])
        #d = d.flat_map(select_areas)
        d = tf.data.Dataset.from_generator(
            load_images,
            output_signature=tf.TensorSpec(shape=(args.img_size, args.img_size, 3), dtype=tf.float32) # type: ignore
        )
        return d.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_dataset(args.batch_size)
    val_dataset = create_dataset(10)
        
    
    #model = VQVAEModel(args)
    model = tf.keras.models.load_model("artifacts\\run_3upcsyi1_model-v1", custom_objects={"VQVAEModel":VQVAEModel, "VectorQuantizer":VectorQuantizer})
    
    wandb_manager = log_and_save.WandbManager("image_outpainting_tokenizer")
    wandb_manager.start(args)
    model_log = ModelLog(model, val_dataset)
    
    model.fit(train_dataset, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_batch_end=model_log.log),
        WandbMetricsLogger(20),
        WandbModelCheckpoint("tokenizer", "reconstruction_loss", save_freq=100)])
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)