from itertools import islice
import argparse
import glob
import re
import random
from pathlib import Path

import tensorflow as tf
from PIL import Image
import numpy as np
from keras.utils import dataset_utils


from log_and_save import WandbLog
from utilities import get_mask_fname, open_image


parser = argparse.ArgumentParser()

parser.add_argument("--dataset_location", default="data", type=str, help="Where data is stored")
parser.add_argument("--places", default=["brno", "belotin", "ceske_budejovice", "cheb"], nargs="+", type=str, help="Individual places to use data from")
parser.add_argument("--example_count", default=5, type=int, help="How many batches of size 8 to log")





class ImageLoading:
    ORDERED_MASKS_FILE = "data/masks.npy"
    FILENAME_DATASET_FILE = "data/filename_dataset.data"


    """Can load the dataset for multiple places and prepare the images for training"""
    def __init__(self, dataset_location, dataset_image_size, places, scale_down = 4, stddev_threshold=0.04, shuffle_buffer=1024):
        """Create a dataset with the given properties"""
        masks, fname_dataset = self.index_dataset(dataset_location)

        #the size to resize images to during loading
        load_image_size = np.array([1200, 1600]) // scale_down

        #create the full image dataset
        self.full_dataset = self._create_dataset(dataset_location, fname_dataset, load_image_size, dataset_image_size, stddev_threshold, shuffle_buffer)

    @classmethod
    def index_dataset(cls, dataset_location):
        mask_fnames = glob.glob("masks/*_mask.png")
        if not Path(cls.ORDERED_MASKS_FILE).exists() or not Path(cls.FILENAME_DATASET_FILE).exists():
            ordered_masks = np.stack([open_image(fname, bool) for fname in mask_fnames], 0)
            np.save(cls.ORDERED_MASKS_FILE, ordered_masks)

            regexp = re.compile("masks/(.*)_mask.png")
            locations = [regexp.match(mask).group(1) for mask in mask_fnames]

            all_images = []

            for mask_i, location in enumerate(locations):
                image_fnames = glob.glob(f"{location}/*/*.jpg", root_dir=dataset_location)
                images_with_indices = [(fname, mask_i) for fname in image_fnames]
                all_images.extend(images_with_indices)
            random.shuffle(all_images)

            #image_fnames, mask_indices = list(zip(*all_images))
            def generator():
                for fname, mask_i in all_images:
                    yield fname, mask_i

            dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))
            dataset.save(cls.FILENAME_DATASET_FILE, "GZIP")
        else:
            ordered_masks = np.load(cls.ORDERED_MASKS_FILE)
            dataset = tf.data.Dataset.load(cls.FILENAME_DATASET_FILE)
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
        mask = tf.image.resize
        #go over all positions, keep each where the box doesn't overlap with the mask
        return np.array([
            [y, x]
            for y in range(load_image_size[0] - dataset_image_size)
            for x in range(load_image_size[1] - dataset_image_size)
            if np.sum(mask[y:y+dataset_image_size, x:x+dataset_image_size]) == 0
        ])


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


    def _create_place_dataset(self, dataset_location, place, load_image_size, box_image_size, stddev_threshold):
        """Create the dataset for a given place"""

        #get all possible image subsets that do not overlap with the mask
        data_boxes = tf.constant(self._create_boxes(place, load_image_size, box_image_size))
        assert len(data_boxes) > 0, "No boxes could be generated, mask is too restrictive"


        #select one suitable area at random
        def select_area(img):
            box_i = tf.random.uniform([1], 0, len(data_boxes)-1, tf.int32)[0]
            box_pos = data_boxes[box_i]
            image = img[box_pos[0]:box_pos[0]+box_image_size, box_pos[1]:box_pos[1]+box_image_size] / 255.0
            image.set_shape((box_image_size, box_image_size, 3))
            return image
            #return tf.data.Dataset.from_tensor_slices([img[a:a+box_image_size,b:b+box_image_size] / 255.0 for a, b in data_boxes])

        def delete_dark(img):
            return tf.reduce_mean(img) > 0.2

        #return false if the image is too dark - used to filter out night images, or too monochrome
        def delete_monochrome(img):
            #mean color
            mean = tf.reduce_mean(img, (0, 1))
            stddev = tf.reduce_mean(tf.abs(img - mean))
            return stddev > stddev_threshold

        #return false if the image is too monochrome
        #def delete_monochrome(img):
        #    return tf.reduce_mean(tf.square(img-tf.reduce_mean(img, (1, 2), keepdims=True))) > variance_threshold

        #get the paths of all images in the place directory
        image_paths, _, _ = dataset_utils.index_directory(f"{dataset_location}/{place}", labels=None, formats=(".jpg",))

        #create a dataset - load the image, select a suitable subset, delete dark images, then delete monochrome ones, then shuffle the rest
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        width, height = load_image_size[1], load_image_size[0]
        return dataset\
            .map(lambda x:self.load_image(x, width, height), num_parallel_calls=tf.data.AUTOTUNE)\
            .map(select_area)\
            .prefetch(tf.data.AUTOTUNE)\
            .filter(delete_dark)\
            .filter(delete_monochrome)


    def _create_dataset(self, dataset_location, places, load_image_size, box_image_size, stddev_threshold, shuffle_buffer=1024):
        """Create a dataset for multiple places"""
        #create one dataset for each place
        datasets = [self._create_place_dataset(dataset_location, place, load_image_size, box_image_size, stddev_threshold) for place in places]
        #take one image from each dataset, and repeat until all are exhausted
        dataset = tf.data.Dataset.choose_from_datasets(datasets, tf.data.Dataset.range(len(datasets)).repeat(), stop_on_empty_dataset=False)
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
    dataset = ImageLoading(args.dataset_location, 128, args.places, stddev_threshold=0.04, shuffle_buffer=0).create_dataset(8)

    #log the first `args.example_count` batches to wandb
    WandbLog().wandb_init("image_outpainting_dataset", args)

    #! 30% of images are black
    #! at threshold =0.07, out of the remaining, 18% are not monochrome
    #! 4 datasets -> 800 000 images -> * 0.7 * 0.18 / 16 batch size = only 6300 batches. Will change between epochs due to image bboxes changing
    #lower threhsold should help


    for batch in islice(dataset.as_numpy_iterator(), args.example_count):
        WandbLog().log_images("dataset elements", batch).commit()


if __name__ == "__main__":
    main(parser.parse_args([]))
