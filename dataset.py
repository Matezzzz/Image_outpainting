from itertools import islice
import argparse
import glob
import re
import random
from pathlib import Path
import gzip

import tensorflow as tf
from PIL import Image
import numpy as np

from log_and_save import WandbLog
from utilities import get_mask_fname, open_image, get_dataset_location


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_location", default="", type=str, help="Directory to read data from. If not set, the path in the environment variable IMAGE_OUTPAINTING_DATASET_LOCATION is used instead.")
parser.add_argument("--example_count", default=5, type=int, help="How many batches of size 8 to log")




class ImageLoading:
    ORDERED_MASKS_FILE = "data/masks.npy.gz"
    FILENAME_DATASET_FILE = "data/filename_dataset.data"


    """Can load the dataset for multiple places and prepare the images for training"""
    def __init__(self, dataset_location, dataset_image_size, *, scale_down = 4, stddev_threshold=0.04, shuffle_buffer=1024):
        """Create a dataset with the given properties"""
        dataset_location = get_dataset_location(dataset_location)

        

        #the size to resize images to during loading
        load_image_size = np.array([1200, 1600]) // scale_down

        #create the full image dataset
        self.full_dataset = self._create_dataset(dataset_location, load_image_size, dataset_image_size, stddev_threshold, shuffle_buffer)

    @classmethod
    def _index_dataset(cls, dataset_location):
        """Get masks for all locations and tuple of (associated mask index, filename) for every image in the dataset"""

        #determine which locations to use based on the masks available
        mask_fnames = sorted(glob.glob("masks/*_mask.png"))
        #if indexing hasn't been computed before (or some part is missing)
        if not Path(cls.ORDERED_MASKS_FILE).exists() or not Path(cls.FILENAME_DATASET_FILE).exists():
            #load the masks and save them as a numpy array in alphabetical order
            ordered_masks = np.stack([open_image(fname, bool) for fname in mask_fnames], 0)
            masks_file = gzip.GzipFile(cls.ORDERED_MASKS_FILE, "w")
            np.save(file=masks_file, arr=ordered_masks)
            masks_file.close()

            #find out all active locations (the ones that have a mask)
            regexp = re.compile("masks/(.*)_mask.png")
            locations = [regexp.match(mask).group(1) for mask in mask_fnames]

            #find out which images are available for each location, save them and the index of the associated mask
            all_images = []
            for mask_i, location in enumerate(locations):
                image_fnames = glob.glob(f"{location}/*/*.jpg", root_dir=dataset_location)
                images_with_indices = [(fname, mask_i) for fname in image_fnames]
                all_images.extend(images_with_indices)
            #shuffle the order of all images in the dataset
            random.shuffle(all_images)

            #save all image filenames as a gzip-compressed tensorflow dataset
            def generator():
                for fname, mask_i in all_images:
                    yield fname, mask_i
            dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))
            dataset.save(cls.FILENAME_DATASET_FILE, "GZIP")

        #we load everything from file just in case. Dataset cardinality can be computed correctly in this case, and it makes the behaviour the same as all subsequent runs
        masks_file = gzip.GzipFile(cls.ORDERED_MASKS_FILE, 'r')
        ordered_masks = np.load(file=masks_file)
        dataset = tf.data.Dataset.load(cls.FILENAME_DATASET_FILE, compression="GZIP")
        return ordered_masks, dataset

    @staticmethod
    def _get_mask(place, load_image_size):
        """Load a mask for the given place. Print an error if it is not available."""
        try:
            img = Image.open(get_mask_fname(place))
        except FileNotFoundError:
            print ("Mask could not be loaded. Create it for the given place using the segmentation.py file.")
            raise
        #convert the mask to a numpy array
        return np.asarray(img.resize((load_image_size[1], load_image_size[0])))

    @classmethod
    def _create_boxes(cls, mask, load_image_size, dataset_image_size):
        """Create image boxes that do not overlap the mask"""
        #go over all positions, keep each where the box doesn't overlap with the mask
        boxes = np.array([
            [y, x]
            for y in range(load_image_size[0] - dataset_image_size)
            for x in range(load_image_size[1] - dataset_image_size)
            if np.sum(mask[y:y+dataset_image_size, x:x+dataset_image_size]) == 0
        ])
        return boxes


    @staticmethod
    def load_image(fname, width, height):
        """Load an image with the given filename, width and height"""
        #read the file and interpret the image data
        img_data = tf.io.read_file(fname)
        img = tf.image.decode_image(img_data, 3, expand_animations=False)
        #resize the loaded image to target width and height
        img = tf.image.resize(img, (height, width))
        img.set_shape((height, width, 3))
        return img


    def _create_dataset_internal(self, dataset_location, fname_dataset, data_boxes, load_image_size, box_image_size, stddev_threshold):
        """Create the dataset for a given place"""

        width, height = load_image_size[1], load_image_size[0]

        #load the image from a filename and select one random suitable area
        def open_and_select_area(img_fname, mask_index):
            #get the boxes that do not overlap with a mask for the given location
            boxes = data_boxes[mask_index]
            box_count = tf.shape(boxes)[0]
            #if there are no boxes to choose from, return pure black - this will get filtered out in the delete_dark step
            if box_count == 0:
                return tf.zeros([box_image_size, box_image_size, 3])

            #load the image from the file
            img = self.load_image(tf.strings.join([dataset_location, img_fname], separator="/"), width, height)
            #select a box to select from the image
            box_pos = boxes[tf.random.uniform([], maxval=box_count, dtype=tf.int32)]

            #return the suitable area
            image = img[box_pos[0]:box_pos[0]+box_image_size, box_pos[1]:box_pos[1]+box_image_size] / 255.0
            image.set_shape((box_image_size, box_image_size, 3))
            return image

        def delete_dark(img):
            return tf.reduce_mean(img) > 0.2

        def delete_monochrome(img):
            #mean color
            mean = tf.reduce_mean(img, (0, 1))
            #average standard deviation
            stddev = tf.reduce_mean(tf.abs(img - mean))
            #keep all interesting images + 10% of the monochrome ones
            return tf.logical_or(stddev > stddev_threshold, tf.random.uniform([]) < 0.1)

        #go over all available images, select a random area from them, remove all dark images and most monochrome ones
        return fname_dataset\
            .map(open_and_select_area, num_parallel_calls=tf.data.AUTOTUNE)\
            .prefetch(tf.data.AUTOTUNE)\
            .filter(delete_dark)\
            .filter(delete_monochrome)


    def _create_dataset(self, dataset_location, load_image_size, box_image_size, stddev_threshold, shuffle_buffer=1024):
        """Create a dataset for multiple places"""

        #load all masks and image filenames
        masks, fname_dataset = self._index_dataset(dataset_location)

        print (f"Running with a dataset of {fname_dataset.cardinality()} images")

        #resize each mask to the load_image_size
        masks = [np.asarray(Image.fromarray(mask).resize((load_image_size[1], load_image_size[0]))) for mask in masks]
        #generate suitable boxes for each mask, and convert them to a tf.constant
        boxes = tf.ragged.constant([self._create_boxes(mask, load_image_size, box_image_size) for mask in masks])
        #create the full dataset
        dataset = self._create_dataset_internal(dataset_location, fname_dataset, boxes, load_image_size, box_image_size, stddev_threshold)

        #shuffle if required
        if shuffle_buffer != 0:
            dataset = dataset.shuffle(shuffle_buffer)
        return dataset

    @staticmethod
    def batch(dataset, batch_size):
        #create batches from a given dataset
        return dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

    def create_dataset(self, batch_size):
        """Create an image dataset with the given batch size"""
        return self.batch(self.full_dataset, batch_size)

    def create_train_dev_datasets(self, dev_examples, batch_size):
        """Create a train and development datasets. First `dev_examples` images will go to the dev_dataset, the rest will go to the training one"""
        return self.batch(self.full_dataset.skip(dev_examples), batch_size), self.batch(self.full_dataset.take(dev_examples), batch_size)

    @staticmethod
    def image_mean(img, keepdims=False):
        """Compute a mean of an image"""
        #allows dimensions [..., w, h, channels]
        return tf.reduce_mean(img, (-3, -2), keepdims=keepdims)

    @classmethod
    def image_mean_flat(cls, img):
        """Compute mean color of all images in a batch"""
        return tf.reduce_mean(cls.image_mean(img), 0)

    @classmethod
    def image_variance(cls, img):
        """Compute variance of an image"""
        mean = cls.image_mean(img, keepdims=True)
        return tf.reduce_mean(tf.square(img-mean))

    @classmethod
    def image_variance_flat(cls, img):
        """Compute average variance of all images in a batch"""
        return tf.reduce_mean(cls.image_variance(img), 0)

    @staticmethod
    def analyze_dataset(dataset, func, samples=100):
        """Return a np.array of the first `samples` results of `func` when called on elements from the dataset"""
        return np.array(list(map(func, islice(dataset.as_numpy_iterator(), samples))))

    @classmethod
    def img_dataset_mean(cls, dataset, samples=100):
        """Estimate the mean of an image dataset using the first `samples` elements"""
        return np.mean(cls.analyze_dataset(dataset, cls.image_mean_flat, samples), 0)

    @classmethod
    def img_dataset_variance(cls, dataset, samples=100):
        """Estimate the variance of an image dataset using the first `samples` elements"""
        return np.mean(cls.analyze_dataset(dataset, cls.image_variance_flat, samples), 0)

    # Deprecated, used for plotting variance distribution of an image dataset
    # @classmethod
    # def plot_image_variances(cls, dataset, samples=1000):
    #     import matplotlib.pyplot as plt
    #     variances = cls.analyze_dataset(dataset, cls.image_variance_flat, samples).ravel()
    #     plt.hist(variances, bins=100)
    #     plt.show()




def main(args):
    #create a default dataset
    dataset = ImageLoading(args.dataset_location, 128, stddev_threshold=0.04, shuffle_buffer=0).create_dataset(8)

    #log the first `args.example_count` batches to wandb
    WandbLog().wandb_init("image_outpainting_dataset", args)
    for batch in islice(dataset.as_numpy_iterator(), args.example_count):
        WandbLog().log_images("dataset elements", batch).commit()


if __name__ == "__main__":
    main(parser.parse_args([]))
