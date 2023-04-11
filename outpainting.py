import tensorflow as tf
from utilities import get_maskgit_fname, get_tokenizer_fname, tf_init, get_sharpening_fname
from maskgit import MaskGIT, MASK_TOKEN
#from tokenizer import VQVAEModel
from dataset import ImageLoading
from itertools import islice
from log_and_save import WandbManager, Log
import wandb
from diffusion_model_upscale import DiffusionModel


import argparse

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", default=0, type=int, help="Which GPU to use. -1 to run on CPU.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")


parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--attempt_count", default=4, type=int, help="How many times to repeat outpainting for one input image.")
parser.add_argument("--example_count", default=10, type=int, help="How many times to do the outpainting on a batch")
parser.add_argument("--outpaint_range",default=2, type=int, help="How many times to outpaint in each direction")
parser.add_argument("--generation_temp",default=1.0, type=float, help="How random should the generation be. Not used for simple decoding.")
parser.add_argument("--samples", default=4, type=int, help="Rendering samples")
parser.add_argument("--decoding",default="full", type=str, help="What decoding method to use, simple/full")

parser.add_argument("--maskgit_steps", default=12, type=int, help="Steps during maskgit generation")
parser.add_argument("--diffusion_steps", default=50, type=int, help="Steps during diffusion model upscaling")

parser.add_argument("--generate_upsampled", default=True, type=bool, help="Whether to generate upscaled images")

parser.add_argument("--maskgit_run", default="025", type=str, help="The maskgit version to use")
parser.add_argument("--sharpen_run", default="", type=str, help="The sharpener/upscaler version to use")\

parser.add_argument("--outpainting_step", default=0.25, type=float, help="The step to use during token generation and upsampling")



parser.add_argument("--dataset_location", default="data", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")







class ProgressBar:
    def __init__(self, generator, total_steps = None, print_steps = 30):
        if total_steps is None:
            total_steps = len(generator)
        self.total_steps = total_steps
        self.print_steps = print_steps
        self.steps = 0
        self.i = 0
        self.generator = generator
        
    def step(self):
        self.steps += 1
        new_i = (self.steps + 0.5) / self.total_steps * self.print_steps
        c = int(new_i - self.i)
        if c:
            print ("#"*c, flush=True, end="")
        self.i += c
    
    def __iter__(self):
        self.start()
        for a in self.generator:
            yield a
            self.step()
        self.end()
        
    def start(self):
        print (f"|{' '*self.print_steps}|" + '\b'*(self.print_steps+1), end="", flush=True)
        
    def end(self):
        print("|", end="\n")





class SpiralGenerator:
    def __init__(self, spiral_layers, spiral_middle, step_size, start_radius = None, include_mid = False):
        self.layers = spiral_layers
        self.middle = spiral_middle
        self.step = step_size
        self.start_radius = start_radius if start_radius is not None else step_size
        self.include_mid = include_mid
        
    def __iter__(self):
        return self.generate()
    
    def generate(self):
        if self.include_mid:
            yield self.middle
        for outpaint_step in range(self.layers):
            hs = (outpaint_step + 1) * self.step
            corners = np.array([self.middle-hs, self.middle+np.array([-hs[0], hs[1]]), self.middle+hs, self.middle+np.array([hs[0], -hs[1]])])
            side_steps = 2*(outpaint_step+1)
            def get_pos(i):
                ci, si = divmod(i, side_steps)
                c1, c2 = corners[ci], corners[(ci+1)%4]
                return c1 + (c2 - c1) // side_steps * si 
            for sub_step in range(4*side_steps):
                yield get_pos((sub_step+1)%(4*side_steps))
         
    def __len__(self):
        return sum([4*(2*(s+1)) for s in range(self.layers)])


class GridGenerator:
    def __init__(self, max_pos, step_count):
        self.max_pos = max_pos
        self.step_count = step_count
        
    def __iter__(self):
        return self.generate()
    
    def generate(self):
        lspace = np.linspace(0, self.max_pos, self.step_count, dtype=np.int32)
        xs, ys = np.meshgrid(lspace[:, 0], lspace[:, 1])
        for a in zip(xs.ravel(), ys.ravel()):
            yield np.array(a)

    def __len__(self):
        return self.step_count * self.step_count



def main(args):
    tf_init(args.use_gpu, args.threads, args.seed)


    maskgit = MaskGIT.load(get_maskgit_fname())
    maskgit.set_generation_temp(args.generation_temp)
    
    upscaler = DiffusionModel.load(get_sharpening_fname())

    dataset = ImageLoading(args.dataset_location, args.img_size, args.places).create_dataset(args.batch_size)
    
    WandbManager("image_outpainting_results").start(args)
    
    
    i_size = args.img_size
    token_mult = np.array(maskgit.downscale_multiplier)
    i_size_tokens = i_size // token_mult
    op_step = (i_size_tokens * args.outpainting_step).astype(np.int32)
    op_total = op_step * args.outpaint_range
    target_size = 2 * op_total + i_size_tokens
    mid = op_total + i_size_tokens//2
    
    # for pred_pos in SpiralGenerator(args.outpaint_range, mid, op_step):
    #     f, t =  pred_pos- i_size_tokens//2, pred_pos+i_size_tokens//2
    #     mask = np.zeros(target_size)
    #     mask[op_total[0]:op_total[0]+i_size_tokens[0], op_total[1]:op_total[1]+i_size_tokens[1]] = 0.5
    #     mask[f[0]:t[0], f[1]:t[1]] = 1.0
    #     Log().log_image("generation mask", mask).commit()
    
    
    #imgs = np.tile(np.arange(128)[:, np.newaxis, np.newaxis], (1, 32, 32))
    #dec = maskgit.from_tokens(imgs)
    #Log().log_images("tokens visualized", dec).commit()
    
    
    for batch_i, batch in islice(enumerate(iter(dataset)), args.example_count):
        batch_size = tf.shape(batch)[0].numpy()
        batch_repeated = tf.repeat(batch, tf.fill(batch_size, args.attempt_count), axis=0)
        
        
        print (f"Batch {batch_i+1} out of {args.example_count}:")
        initial_tokens = maskgit.to_tokens(batch_repeated)
        tokens = np.full([batch_size*args.attempt_count, target_size[0], target_size[1]], MASK_TOKEN, dtype=np.int32)
        tokens[:, op_total[0]:op_total[0]+i_size_tokens[0], op_total[1]:op_total[1]+i_size_tokens[1]] = initial_tokens

        img_dark = tf.reduce_min(tf.reduce_mean(batch_repeated, -1), (1, 2))
           
        print(f" * {'Generating tokens:': <20} ", end="", flush=True)
        for pred_pos in ProgressBar(SpiralGenerator(args.outpaint_range, mid, op_step)):
            
            f, t = pred_pos- i_size_tokens//2, pred_pos+i_size_tokens//2
            inp = tokens[:, f[0]:t[0], f[1]:t[1]]
            best_scores, best_tokens = None, None
            
            for _ in range(1):
                if args.decoding == "simple":
                    new_tokens = maskgit.test_decode_simple(inp)[0]
                elif args.decoding == "full":
                    new_tokens = maskgit.test_decode(inp, args.maskgit_steps)[0]
                else:
                    raise NotImplementedError()

                decoded = maskgit.from_tokens(new_tokens)
                
                has_black = tf.reduce_min(tf.reduce_mean(decoded, -1), (1, 2)) < img_dark - 0.1
                color_std_dev = tf.reduce_mean(tf.math.reduce_std(decoded, (1, 2)), -1)
                scores = color_std_dev - tf.where(has_black, 1.0, 0.0)
                #decoding_valid = tf.logical_and(not_black, not_monochrome)
                
                if best_tokens is None:
                    best_tokens = new_tokens
                    best_scores = scores
                else:
                    best_tokens = tf.where((scores > best_scores)[:, tf.newaxis, tf.newaxis], new_tokens, best_tokens)
                    best_scores = tf.maximum(best_scores, scores)
                
                
                #if tf.reduce_any(done_now):
                #    tokens[done_now, f[0]:t[0], f[1]:t[1]] = np.where(tokens[done_now, f[0]:t[0], f[1]:t[1]] == MASK_TOKEN, new_tokens[done_now], tokens[done_now, f[0]:t[0], f[1]:t[1]])
                
                #done = tf.logical_or(done, done_now)
                #if tf.reduce_all(done):
                #    break
            #if not tf.reduce_all(done):
            #    print ("Not all images could be generated conditions!")
            #print (f"{current_step+1}/{total_steps} done.")
            #current_step += 1
            tokens[:, f[0]:t[0], f[1]:t[1]] = np.where(tokens[:, f[0]:t[0], f[1]:t[1]] == MASK_TOKEN, best_tokens, tokens[:, f[0]:t[0], f[1]:t[1]])


        final_img_lr_size = np.array([tf.shape(batch_repeated)[0], *(token_mult*target_size)])
        output_colors_lowres, overlaps_lowres = np.zeros([*final_img_lr_size, 3]), np.zeros([*final_img_lr_size, 1])
        ry, rx = np.arange(1, i_size+1), np.arange(1, i_size+1)
        blending = np.minimum(
            np.minimum(ry, ry[::-1])[:, np.newaxis],
            np.minimum(rx, rx[::-1])[np.newaxis, :]
        )[:, :, np.newaxis]
        
        
        #lspace = np.linspace(0, 2 * op_step * args.outpaint_range, args.outpaint_range * args.samples, dtype=np.int32)
        #xs, ys = np.meshgrid(lspace[:, 0], lspace[:, 1])
        #total_steps = len(lspace[:, 0]) * len(lspace[:, 1])
        #current_step = 0
        print(f" * {'Rendering:': <20} ", end="", flush=True)
        for f in ProgressBar(GridGenerator(2 * op_step * args.outpaint_range, args.outpaint_range * args.samples)):# zip(xs.ravel(), ys.ravel()), xs.size):# lspace[:, 0]:# range(0, 1 + 2 * args.outpaint_range):
            #for x in lspace[:, 1]:
            #f = np.array([x, y])
            t = f+i_size_tokens
            colors = maskgit.from_tokens(tokens[:, f[0]:t[0], f[1]:t[1]])
            fi, ti = f * token_mult, t*token_mult
            output_colors_lowres[:, fi[0]:ti[0], fi[1]:ti[1]] += colors * blending
            overlaps_lowres[:, fi[0]:ti[0], fi[1]:ti[1]] += blending
            #print (f"{current_step+1}/{total_steps} done.")
            #current_step += 1
        final_images_low_res = output_colors_lowres / overlaps_lowres
        
        
        final_image_size = [final_img_lr_size[0], *(final_img_lr_size[1:3] * 4)]
        final_image = np.zeros([*final_image_size, 3])
        final_image_mask = np.zeros([*final_image_size, 1], bool)
        
        log = Log().log_images("Images", batch).log_images("Outpainted low resolution", final_images_low_res)
        if args.generate_upsampled:
            print(f" * {'Upscaling:': <20} ", end="", flush=True)
            #lspace = np.linspace(0, 2 * op_step * args.outpaint_range, args.outpaint_range * args.samples, dtype=np.int32)
            #xs, ys = np.meshgrid(lspace[:, 0], lspace[:, 1])
            
            
            #for x, y in ProgressBar(SpiralGenerator(args.outpaint_range, final_img_lr_size[1:3] // 2, np.array([i_size, i_size]) // 2, include_mid=True)):
            outpaint_size = 2 * args.outpaint_range * op_step * token_mult
            for f in ProgressBar(GridGenerator(outpaint_size, np.ceil((outpaint_size[0] + i_size) * 1.25 / i_size).astype(np.int32))):
                #fx, fy = np.array([x, y]) - i_size//2
                #tx, ty = np.array([x, y]) + i_size//2
                t = f + i_size
                
                inp_img = final_images_low_res[:, f[0]:t[0], f[1]:t[1]]
                
                upscaled_blurry = tf.image.resize(inp_img, [i_size*4, i_size*4], tf.image.ResizeMethod.BICUBIC)
                
                f2, t2 = 4*f, 4*t
                
                upscaled = upscaler.improve_images(upscaled_blurry, args.diffusion_steps, final_image_mask[:, f2[0]:t2[0], f2[1]:t2[1]], final_image[:, f2[0]:t2[0], f2[1]:t2[1]])
                final_image[:, f2[0]:t2[0], f2[1]:t2[1]] = upscaled
                final_image_mask[:, f2[0]:t2[0], f2[1]:t2[1]] = True
            log.log_images("Outpainted full", final_image)
        log.commit()
        
                
                
                

    





if __name__ == "__main__":
    given_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(given_args)