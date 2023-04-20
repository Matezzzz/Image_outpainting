from pathlib import Path
import argparse

import skimage.measure
import cv2
import numpy as np


from utilities import save_image, get_mask_fname, get_mask_dir, load_images_place, GeneratorProgressBar

from log_and_save import WandbLog


parser = argparse.ArgumentParser()

#arguments are only used when running this file itself (not when it is imported)
parser.add_argument("--dataset_location", default="data", type=str, help="Where data is stored")
parser.add_argument("--place", default="belotin", type=str, help="The place to run segmentation for")



# pylint: disable=no-member
# run a 2d convolution with the given kernel
def convolution_2d(img, kernel):
    return cv2.filter2D(img.astype("float"), -1, kernel)
# pylint: enable=no-member


def dilation(img):
    return convolution_2d(img, np.ones([3, 3])) > 0
def erosion(img):
    return convolution_2d(img, np.ones([3, 3])) == 9
def opening(img):
    return dilation(erosion(img))
def closing(img):
    return erosion(dilation(img))

def region_properties(img):
    """
    Get properties of all regions in an image. Returns a tuple (labeled_image, regions). Uses `label` and `regionprops` methods from `skimage.measure`
    """
    labeled_image = skimage.measure.label(img)
    return labeled_image, skimage.measure.regionprops(labeled_image)


#fill all pixels below a pixel with a given label
def fill_below(img, labels, label):
    #avoid modifying the original
    img = img.copy()
    for x in range(img.shape[1]):
        write = False
        #start at the top, once the correct label is found, start filling in the pixels
        for y in range(img.shape[0]):
            if labels[y, x] == label:
                write=True
            if write:
                img[y, x] = 1
    return img


GRAD_MASK_THRESHOLD = 0.02
def gradient_large_enough(img, gradient_magnitude_threshold=GRAD_MASK_THRESHOLD):
    """
    Return true if the gradient of the image at a given position is larger than the given constant
    """
    gradient_x = convolution_2d(img, np.array([[-1, 1]]))
    gradient_y = convolution_2d(img, np.array([[-1], [1]]))
    #gradient magnitude = mean of r,g,b parts
    gradient_magnitude = np.mean(np.abs(gradient_x) + np.abs(gradient_y), -1)
    return gradient_magnitude > gradient_magnitude_threshold

CLOSING_ITERATIONS = 5
def close_mask(img, closing_iterations=CLOSING_ITERATIONS):
    """
    Perform the closing operation several times on a binary image
    """
    for _ in range(closing_iterations):
        img = closing(img)
    return img

FILL_SMALL_REGION_SIZE = 20000
def fill_small_regions(img, fill_max_size = FILL_SMALL_REGION_SIZE):
    """
    Fill all background regions smaller than `fill_max_size`
    """
    #avoid modifying the original
    img = img.copy()
    labeled_img, regions = region_properties(np.logical_not(img))
    #go over all background regions in the image
    for region in regions:
        #if a region is small enough, fill it
        if region.num_pixels < fill_max_size:
            img[labeled_img == region.label] = 1
    return img

DISCARD_SMALL_REGION_SIZE = 300
def discard_regions(img, discard_max_size=DISCARD_SMALL_REGION_SIZE):
    """
    Discard (fill with 0) all mask regions smaller than `discard_max_size`
    """
    #avoid modifying the original
    img = img.copy()
    labeled_img, regions = region_properties(img)
    for region in regions:
        #if the region is small enough, discard it
        if region.num_pixels < discard_max_size:
            img[labeled_img == region.label] = 0
    return img


def fill_below_image_wide(img):
    """
    Fill everything beyond a region that spans the entire image width
    """
    labeled_img, regions = region_properties(img)
    for region in regions:
        _, box_start_x, _, box_end_x = region.bbox
        # if the region bounding box is as wide as the image, fill all pixels below this region
        if box_end_x - box_start_x == image_width:
            img = fill_below(img, labeled_img, region.label)
    return img


INSTITUTE_LOGO_BBOX = (0, 0, 270, 115)
WEATHER_INFO_TEXT_BBOX = (1350, 0, 1600, 250)
def mask_logos_and_text(img):
    """
    Mask the Czech hydrometeorological institute and current weather info text from the image
    """
    def mask_bbox(image, from_x, from_y, to_x, to_y):
        image[from_y:to_y+1, from_x:to_x+1] = 1
    #avoid modifying the original
    img = img.copy()
    #mask the institute and weather info boundinf boxes
    mask_bbox(img, *INSTITUTE_LOGO_BBOX)
    mask_bbox(img, *WEATHER_INFO_TEXT_BBOX)
    return img




#how many images to use for the segmentation, at most
SEGMENTATION_ITERATIONS = 1000

#width and height of all images, and the created mask
image_height, image_width = 1200, 1600
def image_segmentation(dataset_location, place, log_wandb=False, iterations=SEGMENTATION_ITERATIONS):
    """
    Perform image segmentation for a given place and compute the final segmentation mask
    """

    #create the tarket directory if it doesn't exist
    Path(get_mask_dir(place)).mkdir(exist_ok=True)


    #how many times in a row was this pixel masked out. If the maximum over a run is large enough, the region will be in the final mask
    consecutive_count = np.zeros([image_height, image_width])
    #the largest amount of times a pixel was masked out
    max_consecutive_mask = np.zeros_like(consecutive_count)
    #the total number of times a pixel was masked out
    total_mask_count = np.zeros_like(consecutive_count)

    #start wandb logging if requested
    if log_wandb:
        WandbLog.wandb_init("image_outpainting_segmentation", {"dataset_location":dataset_location, "place":place})

    #load the first SEGMENTATION_ITERATIONS images available for a given place
    loaded_images = load_images_place(dataset_location, place, image_count_limit=iterations)

    print ("Computing image segmentation: ", end="")
    #go over all images
    for i, img in enumerate(GeneratorProgressBar(loaded_images)):
        #compute the mask - first, select places where the image rapidly changes (large gradient)
        gradient_mask = gradient_large_enough(img)
        #perform closing several times
        closed_mask = close_mask(gradient_mask)
        #fill all small empty regions
        filled_mask = fill_small_regions(closed_mask)
        #discard all small filled regions
        discard_mask = discard_regions(filled_mask)
        #mask the weather report text and the logo
        sides_masked_mask = mask_logos_and_text(discard_mask)

        resulting_mask = sides_masked_mask

        #log results to wandb every 10 iterations
        if log_wandb and i % 10 == 0:
            WandbLog().log_image("gradient mask", gradient_mask).log_image("closed mask", closed_mask).log_image("filled mask", filled_mask)\
                .log_image("discard_mask", discard_mask).log_image("sides masked mask", sides_masked_mask).commit()

        #if a pixel is masked this step, increase the consecutive sum, else reset it back to 0
        consecutive_count = np.where(resulting_mask, consecutive_count+1, 0)
        max_consecutive_mask = np.maximum(max_consecutive_mask, consecutive_count)
        total_mask_count += resulting_mask
    #compute the final mask - pixel must have been masked at least 10 frames consecutively, and it must be masked in at least 10% of inputs
    final_mask = np.logical_and(max_consecutive_mask > 5, total_mask_count / len(loaded_images) > 0.1)
    #fill all pixels below any image-wide filled regions
    final_mask_filled = fill_below_image_wide(final_mask)

    #log the final mask and the filled version if logging is on
    if log_wandb:
        WandbLog().log_image("final mask", final_mask).log_image("final mask filled", final_mask_filled).commit()
    #save the final mask for a given place and return it
    save_image(get_mask_fname(place), final_mask_filled)
    return final_mask


def main(args):
    """
    Run image segmentation for given command line arguments
    """
    image_segmentation(args.dataset_location, args.place, True)



if __name__ == "__main__":
    main(parser.parse_args([]))
