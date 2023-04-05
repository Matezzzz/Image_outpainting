from itertools import islice
from pathlib import Path
import argparse

import skimage.measure
import cv2
import numpy as np

import matplotlib.pyplot as plt

from utilities import save_image, open_image, get_mask_fname, get_time_mask_fname, get_mask_dir, load_images_place


parser = argparse.ArgumentParser()


parser.add_argument("--dataset_location", default="data", type=str, help="Where data is stored")
parser.add_argument("--place", default="belotin", type=str, help="The place to run segmentation for")





def dilation(img):
    return cv2.filter2D(img.astype("float"), -1, np.ones([3, 3])) > 0
def erosion(img):
    return cv2.filter2D(img.astype("float"), -1, np.ones([3, 3])) == 9
def opening(img):
    return dilation(erosion(img))
def closing(img):
    return erosion(dilation(img))


def fill_below(img, labels, label):
    for x in range(img.shape[1]):
        write = False
        for y in range(img.shape[0]):
            if labels[y, x] == label:
                write=True
            if write:
                img[y, x] = 1
    return img


SEGMENTATION_ITERATIONS = 1000
image_height, image_width = 1200, 1600
def image_segmentation(dataset_location, place, compute_masks=True):
    img_gen = load_images_place(dataset_location, place)
    
    Path(get_mask_dir(place)).mkdir(exist_ok=True)

    i = 0

    consecutive_sum = np.zeros([image_height, image_width])
    max_sum = np.zeros_like(consecutive_sum)
    occurences = np.zeros_like(consecutive_sum)


    initial_masking_keep_percentage = 0.05
    fill_empty_regions_max_size = 20000
    discard_small_regions_max_size = 300
    for img in islice(img_gen, SEGMENTATION_ITERATIONS):
        if compute_masks:
            #mask = np.random.random([image_height, image_width, 1]) < initial_masking_keep_percentage
            #masked_img = np.where(mask, img, 0)
            #for _ in range(5):
            #    neigh = cv2.filter2D(mask.astype("float"), -1, np.ones([3, 3]))[:, :, np.newaxis]
            #    masked_img = np.where(np.logical_and(np.logical_not(mask), neigh > 0), cv2.filter2D(masked_img, -1, np.ones([3, 3])) / (neigh+1e-10), masked_img)
            #    mask = np.logical_or(mask, neigh > 0)
            grad = np.mean(np.abs(cv2.filter2D(img, -1, np.array([[-1, 1]]))) + np.abs(cv2.filter2D(img, -1, np.array([[-1], [1]]))), -1)
            masked_img = grad > 0.02
            #plt.imshow(grad)
            #plt.show()
            #plt.imshow(masked_img.astype("float"))
            #plt.show()

            #err = pred - patches
            #err = pred - img
            #err = masked_img - img
            #err = np.mean(err * err, 2)
            #print(np.mean(err))


            #mask = np.zeros([*patches.shape[0:3]])
            #base_mask = np.zeros(img.shape[0:2])
            #base_mask[err > 0.0005] = 1
            base_mask = masked_img

            cmask = base_mask.copy()
            for _ in range(5):
                cmask = closing(cmask)

            def fill_small_regions(img):
                labeled_img = skimage.measure.label(np.logical_not(img))
                regions = skimage.measure.regionprops(labeled_img)
                for region in regions:
                    if region.num_pixels < fill_empty_regions_max_size:
                        img[labeled_img == region.label] = 1
                return img

            def discard_regions(img):
                labeled_img = skimage.measure.label(img)
                regions = skimage.measure.regionprops(labeled_img)

                for region in regions:
                    if region.num_pixels < discard_small_regions_max_size:
                        img[labeled_img == region.label] = 0
                for region in regions:
                    a, b, c, d = region.bbox
                    if d - b == image_width:
                        img = fill_below(img, labeled_img, region.label)
                return img

            fmask = fill_small_regions(cmask.copy())
            mask = discard_regions(fmask.copy())

            save_image(get_time_mask_fname(place, i), mask)
        else:
            mask = img
        i+=1
        #wandb.log({"image": log_image(img), "segmentation_mask": log_image(mask), "segmentation":log_segmentation(img, mask), "closed_segmentation":log_segmentation(img, cmask), "fill_segmentation":log_segmentation(img, fmask), "discard_and_fill_segmentation": log_segmentation(img, dmask)})
        print ("#", end="", flush=True)

        consecutive_sum = np.where(mask, consecutive_sum+1, 0)
        max_sum = np.maximum(max_sum, consecutive_sum)
        occurences += mask
    final_mask = np.logical_and(max_sum > 10, occurences / i > 0.3)
    save_image(get_mask_fname(place), final_mask)
    return final_mask


def main(args):
    image_segmentation(args.dataset_location, args.place)



if __name__ == "__main__":
    main(parser.parse_args([]))


def load_from_folder_gen(place):
    for i in range(0, 1000000):
        try:
            yield open_image(get_time_mask_fname(place, i)) > 0
        except FileNotFoundError:
            break
