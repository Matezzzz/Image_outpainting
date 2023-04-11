from itertools import islice

import tensorflow as tf
from PIL import Image
import numpy as np

from keras.utils import dataset_utils

from utilities import get_mask_fname, load_images_place
from segmentation import image_segmentation


class ImageLoading:
    def __init__(self, dataset_location, dataset_image_size, places, scale_down = 4, data_box_count = 10, shuffle_buffer=1024):
        self.dataset_location = dataset_location
        self.places = places
        self.image_size = np.array([1600, 1200]) // scale_down
        self.dataset_image_size = dataset_image_size

        masks = [self.get_mask(place) for place in places]
        self.data_boxes = {}
        for place, mask in zip(places, masks):
            boxes = []
            while len(boxes) < data_box_count:
                x, y = np.random.randint(0, self.image_size - 1 - dataset_image_size, 2)
                if np.sum(mask[x:x+dataset_image_size, y:y+dataset_image_size]) == 0:
                    boxes.append([x, y])
            self.data_boxes[place] = boxes
        self.full_dataset = self._create_dataset(shuffle_buffer)


    def get_mask(self, place):
        try:
            img = Image.open(get_mask_fname(place))
        except FileNotFoundError:
            print ("Mask could not be loaded, recomputing.")
            img = Image.fromarray(image_segmentation(place))
        return np.asarray(img.resize(tuple(self.image_size)))


    @staticmethod
    def load_image(fname, width, height):
        img_data = tf.io.read_file(fname)
        img = tf.image.decode_image(img_data, 3, expand_animations=False)
        img = tf.image.resize(img, (width, height))
        img.set_shape((width, height, 3))
        return img

    @staticmethod
    def image_mean(img, keepdims=False):
        #allows dimensions [..., w, h, channels]
        return tf.reduce_mean(img, (-3, -2), keepdims=keepdims)

    @classmethod
    def image_mean_flat(cls, img):
        return tf.reduce_mean(cls.image_mean(img))

    @classmethod
    def image_variance(cls, img):
        mean = cls.image_mean(img, keepdims=True)
        return tf.reduce_mean(tf.square(img-mean))
    
    @classmethod
    def image_variance_flat(cls, img):
        return tf.reduce_mean(cls.image_variance(img), 0)

    def _create_place_dataset(self, place, shuffle_buffer=1024):
        data_boxes = self.data_boxes[place]
        def select_areas(img):
            return tf.data.Dataset.from_tensor_slices([img[a:a+self.dataset_image_size,b:b+self.dataset_image_size] / 255.0 for a, b in data_boxes])
        def delete_dark(img):
            return tf.reduce_mean(img) > 0.2
        def delete_monochrome(img):
            #return tf.reduce_mean(tf.math.reduce_std(img, (1, 2)), -1) > 0.08
            return tf.reduce_mean(tf.square(img-tf.reduce_mean(img, (1, 2), keepdims=True))) > 0.003
        image_paths, _, _ = dataset_utils.index_directory(f"{self.dataset_location}/{place}", labels=None, formats=(".jpg",))
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        width, height = self.image_size[0], self.image_size[1]
        return dataset\
            .map(lambda x:self.load_image(x, width, height), num_parallel_calls=tf.data.AUTOTUNE)\
            .flat_map(select_areas)\
            .prefetch(tf.data.AUTOTUNE)\
            .filter(delete_dark)\
            .filter(delete_monochrome)\
            .shuffle(shuffle_buffer)

    #def _create_place_dataset(self, place):
        #data_boxes = self.data_boxes[place]
        #dataset = tf.keras.utils.image_dataset_from_directory(f"{self.dataset_location}/{place}", labels=None, batch_size=None, image_size=self.image_size.T    )
        #return dataset.flat_map(select_areas).filter(delete_dark).filter(delete_monochrome).prefetch(tf.data.AUTOTUNE)

    def _create_dataset(self, shuffle_buffer=1024):
        datasets = [self._create_place_dataset(place, shuffle_buffer) for place in self.places]
        dataset = tf.data.Dataset.choose_from_datasets(datasets, tf.data.Dataset.range(len(datasets)).repeat(), stop_on_empty_dataset=False)
        return dataset

    @staticmethod
    def batch(dataset, batch_size):
        return dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)

    def create_dataset(self, batch_size):
        return self.batch(self.full_dataset, batch_size)

    def create_train_dev_datasets(self, dev_examples, batch_size):
        return self.batch(self.full_dataset.skip(dev_examples), batch_size), self.batch(self.full_dataset.take(dev_examples), batch_size)

    @staticmethod
    def analyze_dataset(dataset, func, samples=100):
        return np.array(list(map(func, islice(dataset.as_numpy_iterator(), samples))))

    #dataset elements of form [batch, width, height, channels]
    @classmethod
    def img_dataset_mean(cls, dataset, samples=100):
        return np.mean(cls.analyze_dataset(dataset, cls.image_mean_flat, samples), 0)

    @classmethod
    def img_dataset_variance(cls, dataset, samples=100):
        return np.mean(cls.analyze_dataset(dataset, cls.image_variance_flat, samples), 0)

    @classmethod
    def plot_image_variances(cls, dataset, samples=1000):
        import matplotlib.pyplot as plt
        variances = cls.analyze_dataset(dataset, cls.image_variance_flat, samples).ravel()
        plt.hist(variances, bins=100)
        plt.show()
