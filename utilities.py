import glob
import numpy as np
import wandb
from PIL import Image


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

def save_image(fname, image):
    Image.fromarray(image).save(fname)

def load_images(pattern):
    for fname in glob.glob(pattern):
        yield open_image(fname)

def load_images_place(data_dir, place, day="*"):
    for fname in glob.glob(f"{data_dir}/{place}/{day}/*.jpg"):
        yield open_image(fname)

def get_time_mask_fname(place, time):
    return f"masks/{place}/{time:04d}_mask.png"

def get_mask_fname(place):
    return f"masks/{place}_mask.png"


# import random
# a = glob.glob("brno/*/*.jpg")
# random.shuffle(a)
# print(np.var([np.asarray(Image.open(f))/255.0 for f in a[:100]]))
    

#logger = log_and_save.WandbManager("image_outpainting")
#logger.start({})



#image_segmentation_based(load_images("brno", "20210312"))
# mask = compute_final_mask("masks", "brno")
# save_image(mask, "final_mask.png")
# for img in islice(load_images("brno", "20210312"), 0, 281, 40):
#     wandb.log({"final_mask":log_segmentation(img, mask)})




        
