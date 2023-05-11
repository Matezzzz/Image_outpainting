import glob
import os

import numpy as np
from PIL import Image

#os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default




class LoadImagesGenerator:
    """
    Class for iterating over images in a directory
    """
    def __init__(self, glob_pattern, max_image_count = None):
        """
        Create the generator with the given glob pattern. Limit the number of images if requested.
        """
        self.images = sorted(glob.glob(glob_pattern))
        if max_image_count is not None:
            self.images = self.images[:max_image_count]

    def __len__(self):
        """
        How many images can be loaded
        """
        return len(self.images)

    def __iter__(self):
        """
        Iterate over all images
        """
        for fname in self.images:
            yield open_image(fname)




def get_dataset_location(args_location):
    """Get dataset location - return args_location if specified, the value of the IMAGE_OUTPAINTING_DATASET_LOCATION environment variable otherwise"""
    if args_location == "":
        loc = os.getenv("IMAGE_OUTPAINTING_DATASET_LOCATION")
        #if location is not specified in any way, raise an error
        if loc is None:
            raise RuntimeError("Dataset location not set")
        return loc
    return args_location

def open_image(fname, dtype=float):
    """Open an image and convert it to the given type if needed. Types "float" and "bool" are supported"""
    img = np.asarray(Image.open(fname))
    if dtype == float:
        return img.astype("float") / 255.0
    if dtype == bool:
        return img != 0
    raise NotImplementedError("Other dtypes not supported")

def save_image(fname, image):
    """Save the given image with the given filename"""
    Image.fromarray(image).save(fname)

def save_images(fname, images):
    """Save all images as 'fname_{i}.png, where i are the indices of individual images"""
    for i, img in enumerate(images):
        save_image(f"{fname}_{i}.png", img)

def load_images(pattern):
    """Generator that loads all images matching a given glob pattern"""
    return LoadImagesGenerator(pattern)

def load_images_place(data_dir, place, day="*", time="*", image_count_limit=None):
    """Load images from a given place. Can limit the total amount if requested"""
    return LoadImagesGenerator(f"{data_dir}/{place}/{day}/{time}.jpg", image_count_limit)

def get_mask_fname(place):
    """Get a segmentation mask filename for the given place. This mask will be used when creating datasets for training all models"""
    return f"masks/{place}_mask.png"

def run_name_none(run_name):
    """Return True if run_name is None or "" """
    return run_name is None or run_name == ""

def get_tokenizer_fname(run_name=None):
    """Get the filename of the VQVAE tokenizer model"""
    return "models/tokenizer" + ("" if run_name_none(run_name) else f"_{run_name}")

def get_maskgit_fname(run_name=None):
    """Get the filename of the MaskGIT model"""
    return "models/maskgit" + ("" if run_name_none(run_name) else f"_{run_name}")

def get_sharpening_fname(run_name=None):
    """Get the filename of the upscaling/sharpening diffusion model"""
    return "models/sharpen" + ("" if run_name_none(run_name) else f"_{run_name}")



class ProgressBar:
    """
    Prints a progress bar for a running operation
    """

    def __init__(self, total_steps, print_steps = 30):
        """
        Create a progress bar. Each time the `ProgressBar.step` method is called, progress will be printed.
        
        Step should be called `total_steps` times, and will print a total of `print_steps` progress characteds over the whole run
        
        The progress bar can be used as:
        ```
        with ProgressBar(...) as p:
            for i in range(...):
                p.step()
        ```
        Or, instead of the `with` block, the `start` and `end` methods can be called manually.
        """
        self.total_gen_steps = total_steps
        self.total_print_steps = print_steps
        self.gen_step = 0
        self.print_step = 0

    def step(self):
        """
        Perform one step, and print progress if necessary
        """
        self.gen_step += 1
        #get the current print step - rescale gen_step to the range [0, print_step]
        new_print_step = self.gen_step / self.total_gen_steps * self.total_print_steps

        #if something new should be printed, do so
        self.print_progress(int(new_print_step))

    def print_progress(self, new_print_step):
        """
        Print progress if the print step has changed
        """
        diff = new_print_step - self.print_step
        #print progress if the print step has changed enough
        if diff != 0:
            print ("#"*diff, flush=True, end="")
        self.print_step += diff

    def start(self):
        """
        Start the progress bar - print the starting and ending line
        """
        print (f"|{' '*self.total_print_steps}|" + '\b'*(self.total_print_steps+1), end="", flush=True)

    def end(self):
        """
        End the progress bar - print the final line and a new line
        """
        self.print_progress(self.total_print_steps)
        print("|", end="\n")

    def __enter__(self):
        """
        Start the progress bar
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        End the progress bar
        """
        self.end()





class GeneratorProgressBar(ProgressBar):
    def __init__(self, generator, total_gen_steps = None, print_steps = 30):
        """
        Create a progress bar for a given generator. Each time an element is taken from the generator `ProgressBar(generator)`, progress will be printed.
        
        If total steps is None, generator must have a `__len__` function, denoting the number of elements it will produce
        """
        #if total_steps is None, try to get the generator length
        super().__init__(total_gen_steps if total_gen_steps is not None else len(generator), print_steps)
        self.generator = generator

    def __iter__(self):
        """
        Produce elements from the generator while printing progress
        """
        self.start()
        for x in self.generator:
            yield x
            self.step()
        self.end()
