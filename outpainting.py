from itertools import islice
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


from utilities import get_maskgit_fname, get_sharpening_fname, GeneratorProgressBar, save_images
from tf_utilities import tf_init
from maskgit import MaskGIT, MASK_TOKEN
from dataset import ImageLoading
from log_and_save import WandbLog
from diffusion_model_upscale import DiffusionModel




parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")


parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--attempt_count", default=4, type=int, help="How many times to repeat outpainting for one input image.")
parser.add_argument("--example_count", default=6, type=int, help="How many times to do the outpainting on a batch")
parser.add_argument("--outpaint_range",default=1, type=int, help="How many times to outpaint in each direction")
parser.add_argument("--generation_temp",default=4.0, type=float, help="How random should the generation be. Not used for simple decoding.")
parser.add_argument("--samples", default=2, type=int, help="Rendering samples")
parser.add_argument("--decoding",default="full", type=str, help="What decoding method to use, simple/full")

parser.add_argument("--maskgit_steps", default=12, type=int, help="Steps during maskgit generation")
parser.add_argument("--diffusion_steps", default=50, type=int, help="Steps during diffusion model upscaling")

parser.add_argument("--sides_only", default=True, type=bool, help="Whether to generate only to the sides or in all directions")
parser.add_argument("--generate_upsampled", default=True, type=bool, help="Whether to generate upscaled images")

parser.add_argument("--maskgit_run", default="", type=str, help="The maskgit version to use")
parser.add_argument("--sharpen_run", default="", type=str, help="The sharpener/upscaler version to use")\

parser.add_argument("--outpaint_step", default=0.5, type=float, help="The step to use during token generation and upsampling")



parser.add_argument("--dataset_location", default="outpaint", type=str, help="Directory to read data from. If not set, the path in the environment variable IMAGE_OUTPAINTING_DATASET_LOCATION is used instead.")
parser.add_argument("--dataset_outpaint_only", default=True, type=bool, help="Whether the specified dataset is used only for outpainting (just contains images to expand) or whether it contains raw data")





class SpiralGenerator:
    """
    A generator class for points on a square spiral
    """
    def __init__(self, spiral_layers, spiral_middle, step_size, include_mid = False):
        """
        Create a spiral generator.
        
        :param spiral_layers how many times to go around the spiral, all 4 sides
        :param spiral_middle coordinates of the spiral middle point
        :param step_size the radius of the first spiral square, and the step between two different rectangles
        :param include_mid whether to return the middle point before starting to circle around
        """
        self.layers = spiral_layers
        self.middle = spiral_middle
        self.step = step_size
        self.include_mid = include_mid

    def __iter__(self):
        #return the middle if enabled
        if self.include_mid:
            yield self.middle

        #go over all squares that the spiral is composed of
        for outpaint_step in range(self.layers):
            #compute how many points will be on one side of the square
            step_count = 2*(outpaint_step+1)
            #go over all positions on the square and return them
            for pos in self._generate_square(step_count):
                yield pos

    def _generate_square(self, steps_per_side):
        #find out the length of one side
        side_len = steps_per_side * self.step
        radius = side_len // 2
        #compute square corners
        square_from, square_to = self.middle - radius, self.middle + radius
        corners = np.array([[square_from, square_from], [square_to, square_from], [square_to, square_to], [square_from, square_to]])
        #get position on the square
        def get_pos(i):
            #compute the index of the corner before the point and the index on the current side
            corner_i, side_i = divmod(i, steps_per_side)
            #compute the corners between which the point lies
            corner_from, corner_to = corners[corner_i], corners[(corner_i+1)%4]
            #interpolate between the two corners based on side_i
            return corner_from + (corner_to - corner_from) * side_i // steps_per_side
        #go over all four square sides
        for sub_step in range(4*steps_per_side):
            yield get_pos((sub_step+1)%(4*steps_per_side))

    def __len__(self):
        """
        Compute number of spiral points generated by this class
        """
        # = sum of all squares, 4 sides per square, 2*(s+1) points on each side, where s is the distance from center
        return sum(4*(2*(s+1)) for s in range(self.layers))


class GridGenerator:
    """
    Class that generates all points on a grid. Think np.linspace in 2D
    """
    def __init__(self, max_pos, step_count):
        """
        Create a grid generator. Points will lie between 0 and max_pos, step_count samples will be created in each dimension.
        """
        self.max_pos = max_pos
        self.step_count = step_count

    def __iter__(self):
        """
        Iterate over the grid defined by this class
        """
        #create the 1D linspaces for both x and y
        lspace = np.linspace(0, self.max_pos, self.step_count, dtype=np.int32)
        #coordinates for all x and y points
        x_positions, y_positions = np.meshgrid(lspace, lspace)
        #go over all coordinates, return a numpy array of each double separately
        for position in zip(x_positions.ravel(), y_positions.ravel()):
            yield np.array(position)

    def __len__(self):
        """
        Get number of points on this grid = step_count^2
        """
        return self.step_count * self.step_count




# pylint: disable=too-few-public-methods
class OutpaintingInfo:
    """
    Holds constants that are used during many outpainting methods.
    """

    def __init__(self, args : argparse.Namespace, maskgit : MaskGIT):
        self.image_size = args.img_size
        self.sides_only = args.sides_only

        self.outpaint_range = args.outpaint_range

        self.pixels_per_token = maskgit.downscale_multiplier

        self.image_size_tokens = self.image_size // self.pixels_per_token
        self.outpaint_step_tokens = int(self.image_size_tokens * args.outpaint_step)
        self.outpainting_total_tokens = self.outpaint_step_tokens * args.outpaint_range
        self.outpainted_image_size_tokens = 2 * self.outpainting_total_tokens + self.image_size_tokens
        self.outpainted_image_middle_tokens = self.outpainted_image_size_tokens//2

        self.outpainting_total = self.outpainting_total_tokens * self.pixels_per_token
        self.outpainted_image_size = self.outpainted_image_size_tokens * self.pixels_per_token

        #size of the original image, if it got upscaled
        self.upscaled_image_size = 4 * self.image_size

        #size of the upscaled & outpainted image
        self.outpainted_upscaled_image_size = 4 * self.outpainted_image_size

        self.samples = args.samples

    @property
    def known_tokens_pos(self):
        """Return the middle of the outpainted tokens"""
        return np.array([0 if self.sides_only else self.outpainting_total_tokens, self.outpainting_total_tokens])

    def token_outpaint_generator(self):
        """The generator to use when outpainting - returns middles of all outpainting positions"""
        if self.sides_only:
            #go on a line at the center, back and forth
            return np.array([
                [self.image_size_tokens // 2, self.outpainted_image_middle_tokens - k * (i+1) * self.outpaint_step_tokens]
                for k in [-1, 1] for i in range(self.outpaint_range)
            ])
        return SpiralGenerator(self.outpaint_range, self.outpainted_image_middle_tokens, self.outpaint_step_tokens)

    def decode_generator(self):
        """The generator to use when decoding - either samples on a line or on a grid, for generating sides or all directions respectively"""
        if self.sides_only:
            steps = (self.outpaint_range+1) * self.samples
            return np.stack([
                np.zeros(steps, np.int32),
                np.linspace(0, 2 * self.outpainting_total_tokens, steps, dtype=np.int32)
            ], 1)
        return GridGenerator(2*self.outpainting_total_tokens, (self.outpaint_range+1) * self.samples)

    def upscale_image_generator(self):
        """The generator to use when upscaling"""
        outpainting_total_both_sides = 2 * self.outpainting_total
        steps = int(np.ceil(self.outpainted_image_size * 1.25 / self.image_size))
        if self.sides_only:
            return np.stack([
                np.zeros(steps, np.int32),
                np.linspace(0, outpainting_total_both_sides, steps, dtype=np.int32)
            ], 1)
        return GridGenerator(outpainting_total_both_sides, steps)

    @property
    def outpainted_image_height_tokens(self):
        return self.image_size_tokens if self.sides_only else self.outpainted_image_size_tokens

    @property
    def outpainted_image_height(self):
        return self.image_size if self.sides_only else self.outpainted_image_size

    @property
    def outpainted_upscaled_image_height(self):
        return self.upscaled_image_size if self.sides_only else self.outpainted_upscaled_image_size


def get_slice(data, data_from, data_size):
    """A shortcut for `data[:, data_from[0]:data_from[0]+data_size[0], data_from[1]:data_from[1]+data_size[1]]`"""
    def prep(x):
        return (x, x) if isinstance(x, int) else x
    data_to = data_from + data_size
    data_from, data_to = prep(data_from), prep(data_to)
    return data[:, data_from[0]:data_to[0], data_from[1]:data_to[1]]


def set_slice(data, data_from, data_size, set_data, operation = None):
    """
    A shortcut for `data[:, data_from[0]:data_from[0]+data_size[0], data_from[1]:data_from[1]+data_size[1]] = set_data`
    
    operation can be passed to perform a different operation than assignment.
    """
    data_slice = get_slice(data, data_from, data_size)
    if operation is None:
        data_slice[...] = set_data
    else:
        operation(data_slice, set_data)


def outpaint_tokens(source_tokens, maskgit : MaskGIT, info : OutpaintingInfo, decoding_steps : int, generation_temperature : float, decoding="full"):
    """
    Outpaint given tokens using the provided MaskGIT model.

    Decoding can be either "full" or "simple".
    """

    #the array we will fill with the outpainted tokens
    outpainted_tokens = np.full([source_tokens.shape[0], info.outpainted_image_height_tokens, info.outpainted_image_size_tokens], MASK_TOKEN, dtype=np.int32)
    #save the tokens of the original image at the center
    set_slice(outpainted_tokens, info.known_tokens_pos, info.image_size_tokens, source_tokens)

    token_mask = np.ones([128], bool)
    banned_tokens = [64, 68, 71, 80, 95, 98]
    token_mask[banned_tokens] = False
    token_mask = tf.constant(token_mask)

    #start near the middle, where the original tokens are placed, and gradually continue outwards
    for pred_pos in GeneratorProgressBar(info.token_outpaint_generator()):
        outpaint_from = pred_pos - info.image_size_tokens//2
        #get tokens that serve as current input to maskgit
        input_tokens = get_slice(outpainted_tokens, outpaint_from, info.image_size_tokens)

        #we try multiple times and take the result with the largest variance. This should cause more interesting images to get chosen.
        best_scores, best_tokens = None, None
        for _ in range(1):
            #figure out the current missing tokens
            if decoding == "simple":
                new_tokens = maskgit.test_decode_simple(input_tokens)[0]
            elif decoding == "full":
                new_tokens = maskgit.test_decode(input_tokens, decoding_steps, generation_temperature, token_mask)[0]
            else:
                raise NotImplementedError()

            #get the image back from decoded tokens
            decoded = maskgit.from_tokens(new_tokens)

            #has_black = tf.reduce_min(tf.reduce_mean(decoded, -1), (1, 2)) < img_dark - 0.1
            #use the image to compute its standard deviation - a measure of how diverse an image is - we prefer images with more changes in color
            color_std_dev = tf.reduce_mean(tf.math.reduce_std(decoded, (1, 2)), -1)
            scores = color_std_dev# - tf.where(has_black, 1.0, 0.0)

            #keep the tokens with the largest score
            if best_tokens is None:
                best_tokens = new_tokens
                best_scores = scores
            else:
                best_tokens = tf.where((scores > best_scores)[:, tf.newaxis, tf.newaxis], new_tokens, best_tokens)
                best_scores = tf.maximum(best_scores, scores)
        #replace all undefined tokens with the best tokens instead

        input_tokens[...] = np.where(input_tokens == MASK_TOKEN, best_tokens, input_tokens)

        #set_slice(outpainted_tokens, outpaint_from, info.image_size_tokens, np.where(input_tokens == MASK_TOKEN, best_tokens, input_tokens))
    #print (f"Result: {np.mean(outpainted_tokens==MASK_TOKEN)}")
    return outpainted_tokens


def decode_outpainted_tokens(outpainted_tokens, maskgit : MaskGIT, info : OutpaintingInfo):
    """
    Convert the outpainted tokens to an image
    """

    #size of the image after decoding
    decoded_size = np.array([tf.shape(outpainted_tokens)[0], info.outpainted_image_height, info.outpainted_image_size])
    #output image = sum of rendering results, weighed by blending weights
    output_image = np.zeros([*decoded_size, 3])
    #blending sum = sum of blending weights for each pixel
    blending_sum = np.zeros([*decoded_size, 1])

    blend_range = np.arange(1, info.image_size+1)
    #blending value = distance from the border. Used to mix multiple samples during rendering
    blending = np.minimum(blend_range, blend_range[::-1])[np.newaxis, :]
    if not info.sides_only:
        blending = np.minimum(blending, np.minimum(blend_range, blend_range[::-1])[:, np.newaxis],)
    blending = blending[:, :, np.newaxis]

    #for all samples on a grid
    for render_from in GeneratorProgressBar(info.decode_generator()):
        #convert the rectangle at current position to an image
        colors = maskgit.from_tokens(get_slice(outpainted_tokens, render_from, info.image_size_tokens))

        render_output_from = render_from * info.pixels_per_token

        def add(x, y):
            x[...] += y

        #add colors * blending to the respective place in the output image
        set_slice(output_image, render_output_from, info.image_size, colors * blending, add)
        #add blending weights to the same place
        set_slice(blending_sum, render_output_from, info.image_size, blending, add)
    # return output image / total blending sum at each place
    return output_image / blending_sum



def upscale_image(image, upscaling_model : DiffusionModel, info : OutpaintingInfo, diffusion_steps):
    """
    Upscale a given outpainted image.
    """

    #size of the resulting image
    final_image_shape = [tf.shape(image)[0], info.outpainted_upscaled_image_height, info.outpainted_upscaled_image_size]

    #the upscaled image
    final_image = np.zeros([*final_image_shape, 3])
    #True for the pixels that were upscaled already
    final_image_mask = np.zeros([*final_image_shape, 1], bool)

    #go over all positions in the small image
    for upscale_from in GeneratorProgressBar(info.upscale_image_generator()):
        #get the image part
        inp_img = get_slice(image, upscale_from, info.image_size)

        #upscale the decoded image
        upscaled_blurry = tf.image.resize(inp_img, [info.upscaled_image_size, info.upscaled_image_size], tf.image.ResizeMethod.BICUBIC)

        #figure out which parts I should add details to, and which are already done in the target image
        target_image = get_slice(final_image, 4*upscale_from, info.upscaled_image_size)
        target_mask = get_slice(final_image_mask, 4*upscale_from, info.upscaled_image_size)

        #add details to the blurry upscaled images by using the diffusion model
        upscaled = upscaling_model.improve_images(upscaled_blurry, diffusion_steps, target_mask, target_image)

        #save the image with added details, and set the mask to true
        target_image[...] = upscaled
        target_mask[...] = True
    return final_image


def main(args):
    #set up tensorflow to use the specified GPU, number of threads, and seed
    tf_init(args.use_gpu, args.threads, args.seed)

    #load maskgit from a file and set the generation temperature
    maskgit = MaskGIT.load(get_maskgit_fname(args.maskgit_run))

    #load the upscaler if it is required
    upscaling_model = DiffusionModel.load(get_sharpening_fname(args.sharpen_run)) if args.generate_upsampled else None

    #use just a dataset of loaded images if needed
    if args.dataset_outpaint_only:
        dataset = tf.keras.utils.image_dataset_from_directory("outpaint", labels=None, batch_size=args.batch_size,
                                                              image_size=(args.img_size, args.img_size), shuffle=False)
        dataset = dataset.map(lambda x: x / 255.0).as_numpy_iterator()
    else:
        #load the image dataset with a given batch size
        dataset = ImageLoading(args.dataset_location, args.img_size, stddev_threshold=0.1, monochrome_keep=0.0, shuffle_buffer=0).create_dataset(args.batch_size).as_numpy_iterator()

    #start wandb for logging
    WandbLog.wandb_init("image_outpainting_results", args)

    #holds constants used during multiple outpainting methods
    outpaint_info = OutpaintingInfo(args, maskgit)

    # Visualize the outpainting process.
    # for pred_pos in SpiralGenerator(args.outpaint_range, mid, op_step):
    #     f, t =  pred_pos- i_size_tokens//2, pred_pos+i_size_tokens//2
    #     mask = np.zeros(target_size)
    #     mask[op_total[0]:op_total[0]+i_size_tokens[0], op_total[1]:op_total[1]+i_size_tokens[1]] = 0.5
    #     mask[f[0]:t[0], f[1]:t[1]] = 1.0
    #     Log().log_image("generation mask", mask).commit()

    Path("outpaint_results").mkdir(exist_ok=True)

    #go over all batches
    for batch_i, batch in enumerate(islice(dataset, args.example_count)):
        print (f"Batch {batch_i+1} out of {args.example_count}:")

        save_images(f"outpaint_results/source_{batch_i}", (batch*255).astype(np.uint8))

        #convert the batch images to tokens
        initial_tokens = maskgit.to_tokens(batch)

        #repeat each set of tokens (args.attempt_count) times
        initial_tokens_repeated = tf.repeat(initial_tokens, tf.fill(tf.shape(initial_tokens)[0].numpy(), args.attempt_count), axis=0)

        #img_dark = tf.reduce_min(tf.reduce_mean(batch_repeated, -1), (1, 2))

        print(f" * {'Generating tokens:': <20} ", end="", flush=True)
        #generate outpainted tokens
        outpainted_tokens = outpaint_tokens(initial_tokens_repeated, maskgit, outpaint_info, args.maskgit_steps, args.generation_temp, args.decoding)

        print(f" * {'Rendering:': <20} ", end="", flush=True)
        #convert the outpainted tokens back to an image
        outpainted_image = decode_outpainted_tokens(outpainted_tokens, maskgit, outpaint_info)

        #prepare images for being logged to wandb
        log = WandbLog().log_images("Images", batch).log_images("Outpainted low resolution", outpainted_image)

        save_images(f"outpaint_results/outpainted_{batch_i}", (outpainted_image * 255).astype(np.uint8))

        #if upscaling should be done
        if args.generate_upsampled:
            print(f" * {'Upscaling:': <20} ", end="", flush=True)
            #create the final, upscaled image
            final_image = upscale_image(outpainted_image, upscaling_model, outpaint_info, args.diffusion_steps)

            save_images(f"outpaint_results/upscaled_{batch_i}", (final_image * 255).astype(np.uint8))
            log.log_images("Outpainted full", final_image)
        #log all images to wandb
        log.commit()



if __name__ == "__main__":
    given_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(given_args)
