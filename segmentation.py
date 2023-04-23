import argparse
from getpass import getpass
from random import randint
from pathlib import Path
import glob
import re

import skimage.measure
import cv2
import numpy as np
import paramiko

from utilities import save_image, get_mask_fname, load_images_place, GeneratorProgressBar, open_image

from log_and_save import WandbLog


parser = argparse.ArgumentParser()


parser.add_argument("--server", default="dyscalculia.ms.mff.cuni.cz", type=str, help="The server to download data from if not available offline")
parser.add_argument("--username", default="matezzzz", type=str, help="The SSH username to use")
parser.add_argument("--server_path", default="/projects/SkyGAN/webcams/chmi.cz/sky_webcams", type=str, help="The path to download data from")
parser.add_argument("--dataset_location", default="", type=str, help="Directory to read data from. If not set, the path in the environment variable IMAGE_OUTPAINTING_DATASET_LOCATION is used instead.")




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

    label_values = np.array([True] + [r.num_pixels < fill_max_size for r in regions])

    new_img = label_values[labeled_img]
    return new_img

DISCARD_SMALL_REGION_SIZE = 300
def discard_regions(img, discard_max_size=DISCARD_SMALL_REGION_SIZE):
    """
    Discard (fill with 0) all mask regions smaller than `discard_max_size`
    """
    #avoid modifying the original
    img = img.copy()
    labeled_img, regions = region_properties(img)

    label_values = np.array([False] + [r.num_pixels > discard_max_size for r in regions])

    new_img = label_values[labeled_img]
    return new_img


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
SEGMENTATION_ITERATIONS = 100

#width and height of all images, and the created mask
image_height, image_width = 1200, 1600
def image_segmentation(dataset_location, place, day="*", log_verbosity=0, iterations=SEGMENTATION_ITERATIONS):
    """
    Perform image segmentation for a given place and compute the final segmentation mask
    """

    #how many times in a row was this pixel masked out. If the maximum over a run is large enough, the region will be in the final mask
    consecutive_count = np.zeros([image_height, image_width])
    #the largest amount of times a pixel was masked out
    max_consecutive_mask = np.zeros_like(consecutive_count)
    #the total number of times a pixel was masked out
    total_mask_count = np.zeros_like(consecutive_count)

    #load the first SEGMENTATION_ITERATIONS images available for a given place (1* = select only images from hours 12:00 - 19:55, selects mostly images during the day)
    loaded_images = load_images_place(dataset_location, place, day, "1*", image_count_limit=iterations)

    #save the brightest image for visualizing the final map
    brightest_image = None

    print (f"{'Computing segmentation:': <25}", end="")
    #go over all images
    for i, img in enumerate(GeneratorProgressBar(loaded_images)):
        if img.shape[0] != image_height or img.shape[1] != image_width:
            print (f"Image with a wrong shape found, {img.shape[:2]}")
            continue

        #if this image is the brightest of all until now, remember it
        if brightest_image is None or np.mean(img) > np.mean(brightest_image):
            brightest_image = img

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
        if log_verbosity != 0 and i % 10 == 0:
            log = WandbLog()
            if log_verbosity == 2:
                log.log_segmentation("gradient mask", img, gradient_mask).log_segmentation("closed mask", img, closed_mask)\
                    .log_segmentation("filled mask", img, filled_mask).log_segmentation("discard_mask", img, discard_mask)
            log.log_segmentation("final mask", img, sides_masked_mask).commit()

        #if a pixel is masked this step, increase the consecutive sum, else reset it back to 0
        consecutive_count = np.where(resulting_mask, consecutive_count+1, 0)
        max_consecutive_mask = np.maximum(max_consecutive_mask, consecutive_count)
        total_mask_count += resulting_mask

    consecutive_threshold = 5
    percentage_threshold = 20

    #true if pixel is active in enough frames consecutively
    consecutive_mask = max_consecutive_mask > consecutive_threshold
    percentage_mask = total_mask_count / len(loaded_images) * 100 > percentage_threshold

    mask_classes = np.where(consecutive_mask, 2, 0) + np.where(percentage_mask, 1, 0)

    #compute the final mask - pixel must have been masked at least 10 frames consecutively, and it must be masked in at least 10% of inputs
    final_mask = np.logical_and(consecutive_mask, percentage_mask)
    #discard some remaining noise before returning the mask
    final_mask_denoised = discard_regions(final_mask)
    #fill all pixels below any image-wide filled regions
    final_mask_filled = fill_below_image_wide(final_mask_denoised)

    #log the final mask and the filled version if logging is on

    np.save(f"masks/temp/{place}_max_consecutive.npy", max_consecutive_mask)
    np.save(f"masks/temp/{place}_percentage.npy", total_mask_count / len(loaded_images) * 100)


    WandbLog().log_segmentation("final mask", brightest_image, final_mask_denoised).log_segmentation("final mask filled", brightest_image, final_mask_filled)\
        .log_segmentation("classes mask", brightest_image, mask_classes, ["None", "Percentage", "Consecutive", "Percentage & Consecutive"]).commit()

    #save the final mask for a given place and return it
    save_image(get_mask_fname(place), final_mask_filled)


def create_final_mask(consecutive_mask, percentage_mask):
    final_mask = np.logical_and(consecutive_mask, percentage_mask)
    #discard some remaining noise before returning the mask
    final_mask_denoised = discard_regions(final_mask)
    #fill all pixels below any image-wide filled regions
    return fill_below_image_wide(final_mask_denoised)






def finalize_masks():
    locations = [re.compile("masks/temp/(.*)_percentage.npy").match(fname).group(1) for fname in glob.glob("masks/temp/*_percentage.npy")]
    percent_fnames = [f"masks/temp/{location}_percentage.npy" for location in locations]
    consec_fnames = [percent_fname.replace("percentage", "max_consecutive") for percent_fname in percent_fnames]

    threshold_options = [(2, 5), (3, 10), (5, 20), (10, 30), (20, 40), (30, 50)]


    for location, percent_fname, consec_fname in zip(locations, percent_fnames, consec_fnames):
        percent = np.load(percent_fname)
        consec = np.load(consec_fname)
        img = next(glob.iglob(f"data/{location}/*/1**.jpg"))

        for consec_thr, percentage_thr in threshold_options:
            mask = create_final_mask(percent > percentage_thr, consec > consec_thr)
            WandbLog().log_segmentation(location, img, mask).commit()

    print ("Write best mask index for each location:")
    for location, percent_fname, consec_fname in zip(locations, percent_fnames, consec_fnames):
        percent = np.load(percent_fname)
        consec = np.load(consec_fname)
        img = next(glob.iglob(f"data/{location}/*/1**.jpg"))
        best_mask_i = int(input(f"Best mask at {location}: "))
        consec_thr, percentage_thr = threshold_options[best_mask_i]
        mask = create_final_mask(percent > percentage_thr, consec > consec_thr)

        WandbLog().log_segmentation(f"{location} final mask", img, mask).commit()
        save_image(get_mask_fname(location), mask)


def get_path(*args):
    return "/".join(args)



DAY_ITERS = 5
def main(args):
    """
    Run image segmentation for given command line arguments
    """
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.server, username=args.username, password=getpass("SSH Password:"))
    sftp = ssh.open_sftp()
    sftp.chdir(args.server_path)

    #start wandb logging if requested
    WandbLog.wandb_init("image_outpainting_segmentation", args)

    #! lysa hora 1 mask?

    def get_server_path(*args_):
        return get_path(args.server_path, *args_)

    def get_local_path(*args_):
        return get_path(args.dataset_location, *args_)

    def download_file(server_path, local_path):
        if Path(local_path).exists():
            return
        sftp.get(server_path, local_path)

    for place in sorted(sftp.listdir()):
        #ignore valmez, volary because the images have a weird size; slamenka has some weird white images' lysa_hora_2 has a lot of mist, and is not as useful
        if Path(get_mask_fname(place)).exists() or place in ["valmez", "slamenka", "volary", "lysa_hora2"]:
            print (f"Mask for {place} exists, skipping creation.")
            continue
        print (f"Computing mask for {place}")

        days = sftp.listdir(place)
        try_days = [days[randint(0, len(days)-1)] for _ in range(DAY_ITERS)]

        best_score = -np.inf
        best_day = ""
        for day in try_days:
            Path(get_local_path(place, day)).mkdir(exist_ok=True, parents=True)

            time = sorted([t for t in sftp.listdir(get_server_path(place, day)) if t[0] == "1"])[0]

            local_path = get_local_path(place, day, time)
            download_file(get_server_path(place, day, time), local_path)
            score = -np.mean(np.abs(open_image(local_path) - np.array([0, 0, 1])))
            if score > best_score:
                best_day, score = day, best_score

        server_dir = get_server_path(place, best_day)
        print (f"{'Downloading data: ': <25}", end="", flush=True)
        count = 0
        for time in GeneratorProgressBar(sftp.listdir(server_dir)):
            if time[0] != "1" or count == 100:
                continue
            download_file(get_server_path(place, best_day, time), get_local_path(place, best_day, time))
            count += 1

        image_segmentation(args.dataset_location, place, best_day)

    finalize_masks()



if __name__ == "__main__":
    main(parser.parse_args([]))
