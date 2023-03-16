import glob
import numpy as np
from itertools import islice
import log_and_save
import wandb
import tensorflow as tf
from PIL import Image
from build_network import ResidualNetworkBuild as nb
import sklearn.feature_extraction.image
import cv2
import skimage.measure
import numba

tf.config.set_visible_devices([], "GPU")

def log_image(img):
    return wandb.Image(img)

def log_segmentation(img, mask):
    return wandb.Image(img, masks={
        "predictions":{
            "mask_data":mask
        }
    })


def open_image(fname):
    return np.asarray(Image.open(fname)).astype("float") / 255.0

def save_image(image, fname):
    Image.fromarray(image).save(fname)

def load_images(dir_name, day="*"):
    for fname in glob.glob(f"{dir_name}/{day}/*.jpg"):
        yield open_image(fname)


sky_colors = np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.34, 0.55, 0.73], [0.89, 0.89, 0.89], [0.6, 0.65, 0.71]])

na = np.newaxis

def image_segmentation_basic(img_gen):
    for i in img_gen:
        mask = (np.min(np.linalg.norm(sky_colors[:, na, na, :] - i, 2, axis=3), 0) < 0.2).astype("int")
        log_segmentation(i, mask)



size = 6
def get_img_patches(img, patch_count=None):
    return sklearn.feature_extraction.image.extract_patches_2d(img, (size, size), max_patches=patch_count)


def get_gen_patches(img_gen, patch_count = None):
    return np.concatenate([get_img_patches(img, patch_count) for img in img_gen], 0)    

def get_train_data(img_gen, samples):
    patches = get_gen_patches(img_gen, samples)
    x = patches.copy()
    x[:, 1:-1, 1:-1] = 0
    return x, patches



mask_threshold = 0.5
def mask_elements(img):
    m = np.random.random([image_height, image_width, 1]) < mask_threshold
    img_f = tf.image.convert_image_dtype(img, tf.float32)
    return tf.concat([tf.where(m, img_f, 0), 1-m.astype(np.float32)], 2), img_f

image_width, image_height = 1600, 1200
# def get_train_data_full():
#     d = tf.keras.utils.image_dataset_from_directory("brno", labels=None, batch_size=None, image_size=(image_height, image_width))
#     d = d.map(mask_elements)
#     return d


# train = False
# model_name = "segmentation_model.h5"
# if train:
#     train_data_samples = 1000
#     #X, y = get_train_data(load_images("brno", "20210312"), train_data_samples)
#     #X, y = get_train_data_full(islice(load_images("brno"), 200))
#     dataset = None#get_train_data_full().take(500).batch(5)
#     model = nb.inp([None, None, 4]) >> nb.conv2d(10, 3, act="relu") >> nb.conv2d(10, 3, act="relu") >> nb.conv2d(3, 3, act=tf.keras.activations.hard_sigmoid) >> nb.model()
#     model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.MeanSquaredError())
#     model.fit(dataset, epochs=1)
#     model.save(model_name)
# else:
#     pass #model = tf.keras.models.load_model(model_name)


def dilation(img):return cv2.filter2D(img.astype("float"), -1, np.ones([3, 3])) > 0
def erosion(img): return cv2.filter2D(img.astype("float"), -1, np.ones([3, 3])) == 9
def opening(img): return dilation(erosion(img))
def closing(img): return erosion(dilation(img))



@numba.njit
def fill_below(img, labels, label):
    for x in range(img.shape[1]):
        write = False
        for y in range(img.shape[0]):
            if labels[y, x] == label: write=True
            if write: img[y, x] = 1
    return img



def image_segmentation_based(img_gen):
    i = 0
    for img in img_gen:
        #patches = get_img_patches(img)
        #x = patches.copy()
        #x[:, 1:-1, 1:-1] = 0
        keep_percentage = 0.05
        mask = np.random.random([image_height, image_width, 1]) < keep_percentage
        masked_img = np.where(mask, img, 0)
        #pred = model.predict(tf.expand_dims(a, 0))[0]
        for _ in range(5):
            neigh = cv2.filter2D(mask.astype("float"), -1, np.ones([3, 3]))[:, :, na]
            masked_img = np.where(np.logical_and(np.logical_not(mask), neigh > 0), cv2.filter2D(masked_img, -1, np.ones([3, 3])) / (neigh+1e-10), masked_img)
            mask = np.logical_or(mask, neigh > 0)

        #err = pred - patches
        #err = pred - img
        err = masked_img - img
        err = np.mean(err * err, 2)
        #print(np.mean(err))
        
        
        #mask = np.zeros([*patches.shape[0:3]])
        mask = np.zeros(img.shape[0:2])
        mask[err > 0.0005] = 1
        
        cmask = mask.copy()
        for _ in range(5):
            cmask = closing(cmask)
        
        max_fill_size = 20000
        def fill_small_regions(img):
            labeled_img = skimage.measure.label(np.logical_not(img))
            regions = skimage.measure.regionprops(labeled_img)
            for region in regions:
                if region.num_pixels < max_fill_size:
                    img[labeled_img == region.label] = 1
            return img
        
        
        
        discard_size = 300
        def discard_regions(img):
            labeled_img = skimage.measure.label(img)
            regions = skimage.measure.regionprops(labeled_img)
            
            for region in regions:
                if region.num_pixels < discard_size:
                    img[labeled_img == region.label] = 0
            for region in regions:
                a, b, c, d = region.bbox
                if d - b == image_width:
                    img = fill_below(img, labeled_img, region.label)
            return img
        
        
        #print (np.max(err))
        #mask_img = sklearn.feature_extraction.image.reconstruct_from_patches_2d(mask, [h, w])
        #if total_mask is None: total_mask = mask_img
        #else: total_mask += mask_img
        fmask = fill_small_regions(cmask.copy())
        dmask = discard_regions(fmask.copy())
        
        save_image(f"masks/brno_{i:04d}.png", dmask[:, :, na])
        i+=1
        #wandb.log({"image": log_image(img), "segmentation_mask": log_image(mask), "segmentation":log_segmentation(img, mask), "closed_segmentation":log_segmentation(img, cmask), "fill_segmentation":log_segmentation(img, fmask), "discard_and_fill_segmentation": log_segmentation(img, dmask)})
        print ("#", end="", flush=True)
    print()


import random
a = glob.glob("brno/*/*.jpg")
random.shuffle(a)
print(np.var([np.asarray(Image.open(f))/255.0 for f in a[:100]]))
    


def compute_final_mask(dirname, fname):
    consecutive_sum = np.zeros([image_height, image_width])
    max_sum = np.zeros_like(consecutive_sum)
    occurences = np.zeros_like(consecutive_sum)
    for i in range(0, 1000000):
        try:
            img = open_image(f"{dirname}/{fname}_{i:04d}.png") > 0
        except FileNotFoundError:
            occurences /= i
            break
        consecutive_sum = np.where(img, consecutive_sum+1, 0)
        max_sum = np.maximum(max_sum, consecutive_sum)
        occurences += img
        print ("#", end="", flush=True)
    print()
    return np.logical_and(max_sum > 10, occurences > 0.3)
        


logger = log_and_save.WandbManager("image_outpainting")
logger.start({})





#idea - sky -> pixel predicatable from surroundings
#normal locations -> nope

#image_segmentation_based(load_images("brno", "20210312"))
mask = compute_final_mask("masks", "brno")
save_image(mask, "final_mask.png")
for img in islice(load_images("brno", "20210312"), 0, 281, 40):
    wandb.log({"final_mask":log_segmentation(img, mask)})
