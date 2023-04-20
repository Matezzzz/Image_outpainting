from itertools import islice

import tensorflow as tf
from PIL import Image
import numpy as np

from keras.utils import dataset_utils

from utilities import get_mask_fname


class ImageLoading:
    """Can load the dataset for multiple places and prepare the images for training"""
    def __init__(self, dataset_location, dataset_image_size, places, scale_down = 4, shuffle_buffer=1024):
        """Create a dataset with the given properties"""

        #the size to resize images to during loading
        load_image_size = np.array([1600, 1200]) // scale_down

        #create the full image dataset
        self.full_dataset = self._create_dataset(dataset_location, places, load_image_size, dataset_image_size, shuffle_buffer)


    def _get_mask(self, place, load_image_size):
        """Load a mask for the given place. Print an error if it is not available."""
        try:
            img = Image.open(get_mask_fname(place))
        except FileNotFoundError:
            print ("Mask could not be loaded. Create it for the given place using the segmentation.py file.")
            raise
            #img = Image.fromarray(image_segmentation(dataset_location, place))
        #convert the mask to a numpy array
        return np.asarray(img.resize(tuple(load_image_size)))


    def _create_boxes(self, place, load_image_size, dataset_image_size):
        """Create image boxes that do not overlap the mask"""
        mask = self._get_mask(place, load_image_size)
        #go over all positions, keep each where the box doesn't overlap with the mask
        return np.array([
            [y, x]
            for y in range(load_image_size[0] - dataset_image_size)
            for x in range(load_image_size[1] - dataset_image_size)
            if np.sum(mask[x:x+dataset_image_size, y:y+dataset_image_size]) == 0
        ])


    @staticmethod
    def load_image(fname, width, height):
        """Load an image with the given filename, width and height"""
        #read the file and interpret the image data
        img_data = tf.io.read_file(fname)
        img = tf.image.decode_image(img_data, 3, expand_animations=False)
        #resize the loaded image to target width and height
        img = tf.image.resize(img, (width, height))
        img.set_shape((width, height, 3))
        return img


    def _create_place_dataset(self, dataset_location, place, load_image_size, box_image_size, shuffle_buffer=1024):
        """Create the dataset for a given place"""

        #get all possible image subsets that do not overlap with the mask
        data_boxes = self._create_boxes(place, load_image_size, box_image_size)

        #select one suitable area at random
        def select_area(img):
            y, x = data_boxes[np.random.randint(0, len(data_boxes)-1)]
            return img[y:y+box_image_size, x:x+box_image_size] / 255.0
            #return tf.data.Dataset.from_tensor_slices([img[a:a+box_image_size,b:b+box_image_size] / 255.0 for a, b in data_boxes])

        #return false if the image is too dark - used to filter out night images
        def delete_dark(img):
            return tf.reduce_mean(img) > 0.2

        #return false if the image is too monochrome
        def delete_monochrome(img):
            return tf.reduce_mean(tf.square(img-tf.reduce_mean(img, (1, 2), keepdims=True))) > 0.003

        #get the paths of all images in the place directory
        image_paths, _, _ = dataset_utils.index_directory(f"{dataset_location}/{place}", labels=None, formats=(".jpg",))

        #create a dataset - load the image, select a suitable subset, delete dark images, then delete monochrome ones, then shuffle the rest
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        width, height = load_image_size[0], load_image_size[1]
        return dataset\
            .map(lambda x:self.load_image(x, width, height), num_parallel_calls=tf.data.AUTOTUNE)\
            .map(select_area)\
            .prefetch(tf.data.AUTOTUNE)\
            .filter(delete_dark)\
            .filter(delete_monochrome)\
            .shuffle(shuffle_buffer)

    def _create_dataset(self, dataset_location, places, load_image_size, box_image_size, shuffle_buffer=1024):
        """Create a dataset for multiple places"""
        #create one dataset for each place
        datasets = [self._create_place_dataset(dataset_location, place, load_image_size, box_image_size, shuffle_buffer) for place in places]
        #take one image from each dataset, and repeat until all are exhausted
        dataset = tf.data.Dataset.choose_from_datasets(datasets, tf.data.Dataset.range(len(datasets)).repeat(), stop_on_empty_dataset=False)
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
