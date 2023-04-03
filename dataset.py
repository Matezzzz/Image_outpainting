import tensorflow as tf
from PIL import Image
import numpy as np


from utilities import get_mask_fname, load_images_place
from segmentation import image_segmentation


class ImageLoading:
    def __init__(self, dataset_location, dataset_image_size, places, scale_down = 4, data_box_count = 10):
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
        self.full_dataset = self._create_dataset()

    def get_mask(self, place):
        try:
            img = Image.open(get_mask_fname(place))
        except FileNotFoundError:
            print ("Mask could not be loaded, recomputing.")
            img = Image.fromarray(image_segmentation(load_images_place(self.dataset_location, place), place))
        return np.asarray(img.resize(tuple(self.image_size)))

    def _create_place_dataset(self, place):
        data_boxes = self.data_boxes[place]
        dataset = tf.keras.utils.image_dataset_from_directory(f"{self.dataset_location}/{place}", labels=None, batch_size=None, image_size=self.image_size.T) # type: ignore
        def select_areas(img):
            return tf.data.Dataset.from_tensor_slices([img[a:a+self.dataset_image_size,b:b+self.dataset_image_size] / 255.0 for a, b in data_boxes])
        def delete_dark(img):
            return tf.reduce_mean(tf.reduce_mean(img, -1)) > 0.2
        return dataset.flat_map(select_areas).filter(delete_dark).prefetch(tf.data.AUTOTUNE) # type: ignore

    def _create_dataset(self):
        datasets = [self._create_place_dataset(place) for place in self.places]
        return tf.data.Dataset.choose_from_datasets(datasets, tf.data.Dataset.range(len(datasets)).repeat(), stop_on_empty_dataset=False).shuffle(1024)

    def create_dataset(self, batch_size):
        return self.full_dataset.batch(batch_size)

    def create_train_dev_datasets(self, dev_examples, batch_size):
        return self.full_dataset.skip(dev_examples).batch(batch_size), self.full_dataset.take(dev_examples).batch(batch_size)
