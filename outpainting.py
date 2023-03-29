import os
GPU_TO_USE = int(open("gpu_to_use.txt").read().splitlines()[0]) # type: ignore
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if GPU_TO_USE == -1 else str(GPU_TO_USE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import tensorflow as tf
from utilities import get_maskgit_fname, get_tokenizer_fname
from maskgit import MaskGIT, MASK_TOKEN
#from tokenizer import VQVAEModel
from dataset import ImageLoading
from itertools import islice
from log_and_save import WandbManager
import wandb

import argparse

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")


parser.add_argument("--img_size", default=128, type=int, help="Input image size")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--example_count", default=10, type=int, help="How many times to do the outpainting on a batch")
parser.add_argument("--outpaint_range",default=2, type=int, help="How many times to outpaint in each direction")
parser.add_argument("--generation_temp",default=.5, type=float, help="How random should the generation be")
parser.add_argument("--samples", default=4, type=int, help="Rendering samples")

parser.add_argument("--dataset_location", default=".", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")







class ProgressBar:
    def __init__(self, generator, total_steps, print_steps = 30):
        self.total_steps = total_steps
        self.print_steps = print_steps
        self.steps = 0
        self.i = 0
        self.generator = generator
        
    def step(self):
        self.steps += 1
        new_i = (self.steps + 0.5) / self.total_steps * self.print_steps
        c = int(new_i - self.i)
        if c: print ("#"*c, flush=True, end="")
        self.i += c
    
    def __iter__(self):
        self.start()
        for a in self.generator:
            yield a
            self.step()
        self.end()
        
    def start(self):
        print (f"|{' '*self.print_steps}|" + '\b'*(self.print_steps+1), end="")
        
    def end(self):
        print("|", end="\n")






import time
def main(args):
    if GPU_TO_USE == -1: tf.config.set_visible_devices([], "GPU")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


    maskgit = MaskGIT.load(get_maskgit_fname()) # type: ignore
    maskgit.set_generation_temp(args.generation_temp)

    dataset = ImageLoading(args.dataset_location, args.img_size, args.places).create_dataset(args.batch_size)
    
    
    WandbManager("image_outpainting_results").start(args)
    
    i_size = args.img_size
    token_mult = np.array(maskgit.downscale_multiplier)
    i_size_tokens = i_size // token_mult
    op_step = i_size_tokens // 2
    op_total = op_step * args.outpaint_range
    target_size = 2 * op_total + i_size_tokens
    mid = op_total + i_size_tokens//2
    for batch_i, batch in islice(enumerate(iter(dataset)), args.example_count):
        print (f"Batch {batch_i}:")
        initial_tokens = maskgit.to_tokens(batch)
        tokens = np.full([tf.shape(batch)[0].numpy(), target_size[0], target_size[1]], MASK_TOKEN, dtype=np.int32)
        tokens[:, op_total[0]:op_total[0]+i_size_tokens[0], op_total[1]:op_total[1]+i_size_tokens[1]] = initial_tokens
        total_steps = sum([4 * (s+2) for s in range(args.outpaint_range)])
        #current_step = 0
        def position_generation():
            for outpaint_step in range(args.outpaint_range):
                hs = (outpaint_step + 1) * i_size_tokens // 2
                corners = np.array([mid-hs, mid+np.array([-hs[0], hs[1]]), mid+hs, mid+np.array([hs[0], -hs[1]])])
                side_steps = side_steps = outpaint_step + 2
                def get_pos(i):
                    ci, si = divmod(i, side_steps)
                    c1, c2 = corners[ci], corners[(ci+1)%4]
                    return c1 + (c2 - c1) // side_steps * si 
                for sub_step in range(4*side_steps):
                    yield get_pos((sub_step+1)%(4*side_steps))
                
        print(f" * {'Generating tokens:': <20} ", end="", flush=True)
        for pred_pos in ProgressBar(position_generation(), total_steps):
            #for sstep in range(4 * side_steps):
                #print ("#", flush=True)
                
            f, t = pred_pos-op_step, pred_pos+op_step
            new_tokens = maskgit.decode(tokens[:, f[0]:t[0], f[1]:t[1]], False)[0]
            tokens[:, f[0]:t[0], f[1]:t[1]] = np.where(tokens[:, f[0]:t[0], f[1]:t[1]] == MASK_TOKEN, new_tokens, tokens[:, f[0]:t[0], f[1]:t[1]])
            #print (f"{current_step+1}/{total_steps} done.")
            #current_step += 1

        final_img_size = [tf.shape(batch)[0], *(token_mult*target_size)]
        output_colors, overlaps = np.zeros([*final_img_size, 3]), np.zeros([*final_img_size, 1])
        ry, rx = np.arange(1, i_size+1), np.arange(1, i_size+1)
        blending = np.minimum(
            np.minimum(ry, ry[::-1])[:, np.newaxis],
            np.minimum(rx, rx[::-1])[np.newaxis, :]
        )[:, :, np.newaxis]
        
        
        lspace = np.linspace(0, 2 * op_step * args.outpaint_range, args.outpaint_range * args.samples, dtype=np.int32)
        xs, ys = np.meshgrid(lspace[:, 0], lspace[:, 1])
        #total_steps = len(lspace[:, 0]) * len(lspace[:, 1])
        #current_step = 0
        print(f" * {'Rendering:': <20} ", end="", flush=True)
        for x, y in ProgressBar(zip(xs.ravel(), ys.ravel()), xs.size):# lspace[:, 0]:# range(0, 1 + 2 * args.outpaint_range):
            #for x in lspace[:, 1]:
            f = np.array([x, y])
            t = f+i_size_tokens
            colors = maskgit.from_tokens(tokens[:, f[0]:t[0], f[1]:t[1]])
            fi, ti = f * token_mult, t*token_mult
            output_colors[:, fi[0]:ti[0], fi[1]:ti[1]] += colors * blending
            overlaps[:, fi[0]:ti[0], fi[1]:ti[1]] += blending
            #print (f"{current_step+1}/{total_steps} done.")
            #current_step += 1
        final_images = output_colors / overlaps
        wandb.log({"Images": [wandb.Image(i) for i in batch], "Outpainted":[wandb.Image(i) for i in final_images]})
        
                
                
                

    





if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)