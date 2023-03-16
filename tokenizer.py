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
parser.add_argument("--embedding_loss_weight", default=1.0, type=float, help="Scale the embedding loss")
parser.add_argument("--commitment_loss_weight", default=0.25, type=float, help="Scale for the commitment loss (penalize changing embeddings)")
parser.add_argument("--entropy_loss_weight", default=0.05, type=float, help="Scale for the entropy loss")
parser.add_argument("--entropy_loss_temperature", default=0.01, type=float, help="Entropy loss temperature")
parser.add_argument("--data_dir", default="brno", type=str, help="Directory to read data from")
parser.add_argument("--mask_fname", default="final_mask_brno.png", type=str, help="Segmentation mask for a directory")








class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, codebook_size, embedding_dim, embedding_loss_weight, commitment_loss_weight, entropy_loss_weight, entropy_loss_temperature, **kwargs):
        super().__init__(**kwargs)
        #these need to be easily accessed in the training loop
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embedding_loss_weight = embedding_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.codebook = self.add_weight("codebook", [codebook_size, embedding_dim], tf.float32, tf.keras.initializers.RandomUniform())
    
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
        
        _, _, counts = tf.unique_with_counts(tf.reshape(tokens, [-1]))
        self.add_metric(tf.cast(tf.shape(counts)[0], tf.float32), "codebook_keys_used")
        
        embedding_loss = self.embedding_loss_weight * tf.reduce_mean((tf.stop_gradient(embeddings) - inputs)**2)
        commitment_loss = self.commitment_loss_weight * tf.reduce_mean((embeddings - tf.stop_gradient(inputs))**2)
        #move all embeddings to their closest vector
        entropy_loss = self.entropy_loss_weight * tf.reduce_mean(tf.reduce_min(tf.reshape(distances, [-1, self.codebook_size]), 0)) #self.compute_entropy_loss(distances)
        
        self.add_loss(commitment_loss + embedding_loss+ entropy_loss)
        self.add_metric(commitment_loss, "commitment_loss"); self.add_metric(embedding_loss, "embedding_loss"); self.add_metric(entropy_loss, "entropy_loss")
        if training: embeddings = inputs + tf.stop_gradient(embeddings - inputs)
        return embeddings
    
        
    def compute_entropy_loss(self, distances):
        #shape [all_points, codebook_size]
        affinity = tf.reshape(-distances, [-1, tf.shape(distances)[-1]]) / self.entropy_loss_temperature
        probs = tf.nn.softmax(affinity, -1)
        log_probs = tf.nn.log_softmax(affinity+1e-5, -1)
        avg_probs = tf.reduce_mean(probs, 0)
        avg_entropy = -tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-5))
        sample_entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, -1)) # type: ignore
        loss = sample_entropy - avg_entropy
        return loss
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "codebook_size":self.codebook_size,
            "embedding_dim":self.embedding_dim,
            "embedding_loss_weight":self.embedding_loss_weight,
            "commitment_loss_weight":self.commitment_loss_weight,
            "entropy_loss_weight":self.entropy_loss_weight,
            "entropy_loss_temperature":self.entropy_loss_temperature
        })
        return config
    
    
        


def create_encoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim):
    return NetworkOp(lambda x:\
           nb.conv2DDown(filters=10)(x)\
        >> nb.residualDownscaleSequence(filters * np.array(residual_layer_multipliers), num_res_blocks, nb.groupnorm, nb.swish)\
        >> nb.groupnorm()\
        >> nb.swish()\
        >> nb.conv2D(embedding_dim, kernel_size=1)
    )

def create_decoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim):
    return NetworkOp(lambda x:\
           nb.conv2D(filters=embedding_dim)(x)\
        >> nb.residualUpscaleSequence(filters * np.array(residual_layer_multipliers[::-1]), num_res_blocks, nb.groupnorm, nb.swish)\
        >> nb.groupnorm()\
        >> nb.swish()\
        >> nb.conv2DUp(filters=6, activation="relu")\
        >> nb.conv2D(filters=3, activation="hard_sigmoid")
    )



class VQVAEModel:
    def __init__(self, img_size, filters, residual_layer_multipliers, num_res_blocks, embedding_dim, args):
        
        self._quantizer = VectorQuantizer(
            args.codebook_size,
            args.embedding_dim,
            args.embedding_loss_weight,
            args.commitment_loss_weight,
            args.entropy_loss_weight,
            args.entropy_loss_temperature
        )

        #self.codebook_size = args.codebook_size
        inp_shape = [img_size, img_size, 3]
        self._encode_model = nb.inp(inp_shape) >> create_encoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim) >> nb.model()
        self._decode_model = nb.inp(self._encode_model.output_shape[1:]) >> create_decoder(filters, residual_layer_multipliers, num_res_blocks, embedding_dim) >> nb.model()
        self._model = nb.inp(inp_shape) >> TensorflowOp(self._encode_model) >> TensorflowOp(self._quantizer) >> TensorflowOp(self._decode_model) >> nb.model()
        data_variance = 0.0915055541062209
        def scaled_mse_loss(y_true, y_pred): return tf.reduce_mean((y_true-y_pred)**2) / data_variance
        self._model.compile(tf.keras.optimizers.Adam(), scaled_mse_loss, metrics=[tf.keras.metrics.MeanSquaredError("reconstruction_loss")])
        
    def get_model(self):
        return self._model
        
    def generate_samples(self, count):
        tokens = np.random.randint(0, self._quantizer.codebook_size, [count, self._encode_model.output_shape[1], self._encode_model.output_shape[2]])
        embed_tokens = self._quantizer.get_embedding(tokens)
        return self._decode_model.predict(embed_tokens)
    
    def load(self, dirname):
        self._model = tf.keras.models.load_model(dirname, custom_objects={"VectorQuantizer":VectorQuantizer})
        self._encode_model = self._model.get_layer(index=1) #type: ignore
        self._quantizer = self._model.get_layer(index=2) #type: ignore
        self._decode_model = self._model.get_layer(index=3) #type: ignore
        
    
            
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
                data, _ = next(self._dataset)
                reconstructed = self._model.get_model().predict(data)
                wandb.log({
                    "images": [wandb.Image(d) for d in data],
                    "reconstruction":[wandb.Image(d) for d in reconstructed],
                    "generated_images": [wandb.Image(d) for d in generated]
                }, commit=False)
    
    
    
    image_size = np.array([1600, 1200]) // 4
    
    mask = np.asarray(Image.open(args.mask_fname).resize(tuple(image_size)))
    data_boxes = []
    while len(data_boxes) < 10:
        x, y = np.random.randint(0, image_size - 1 - args.img_size, 2)
        if np.sum(mask[x:x+args.img_size, y:y+args.img_size]) == 0:
            data_boxes.append([x, y])
        
    
    image_fnames = glob.glob(f"{args.data_dir}/*/*.jpg")
    random.shuffle(image_fnames)
    def create_dataset(batch_size):
        def load_images():
            for i in image_fnames:
                img = np.asarray(Image.open(i).resize(tuple(image_size))) / 255.0
                if np.min(np.sum(img, -1)) < 0.1: continue
                for a, b in data_boxes:
                    x = img[a:a+args.img_size,b:b+args.img_size]
                    
                    yield x, x
        
        #d = tf.keras.utils.image_dataset_from_directory("brno", labels=None, batch_size=None, image_size=image_size.T) # type: ignore
        #def select_areas(img):
        #    return tf.data.Dataset.from_tensor_slices([img / 255.0 for a, b in data_boxes])
        #d = d.flat_map(select_areas)
        img_spec = tf.TensorSpec(shape=(args.img_size, args.img_size, 3), dtype=tf.float32) # type: ignore
        
        d = tf.data.Dataset.from_generator(
            load_images,
            output_signature=(img_spec, img_spec)
        )
        return d.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_dataset(args.batch_size)
    val_dataset = create_dataset(10)
    
    model = VQVAEModel(args.img_size, args.filters, args.residual_layer_multipliers, args.num_res_blocks, args.embedding_dim, args)
    #model.load("tokenizer")
    
    wandb_manager = log_and_save.WandbManager("image_outpainting_tokenizer")
    wandb_manager.start(args)
    model_log = ModelLog(model, val_dataset)
    
    model.get_model().fit(train_dataset, epochs=args.epochs, callbacks=[ #type: ignore
        tf.keras.callbacks.LambdaCallback(on_batch_end=model_log.log),
        WandbMetricsLogger(50),
        WandbModelCheckpoint("tokenizer", "reconstruction_loss", save_freq=5000)]
    )
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)