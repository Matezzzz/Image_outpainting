import os
GPU_TO_USE = int(open("gpu_to_use.txt").read().splitlines()[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if GPU_TO_USE == -1 else str(GPU_TO_USE)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import tensorflow as tf
from utilities import get_maskgit_fname, get_tokenizer_fname
from maskgit import MaskGIT, MASK_TOKEN
from tokenizer import VQVAEModel
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
parser.add_argument("--outpaint_range",default=5, type=int, help="How many times to outpaint in each direction")

parser.add_argument("--dataset_location", default=".", type=str, help="Directory to read data from")
parser.add_argument("--places", default=["brno"], type=list[str], help="Individual places to use data from")








def main(args):
    if GPU_TO_USE == -1: tf.config.set_visible_devices([], "GPU")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


    tokenizer = VQVAEModel.load(get_tokenizer_fname())
    maskgit = MaskGIT.load(get_maskgit_fname()) # type: ignore

    dataset = ImageLoading(args.dataset_location, args.img_size, args.places).create_dataset(args.batch_size)
    
    
    WandbManager("image_outpainting_results").start(args)
    
    i_size = args.img_size
    token_mult = np.array(tokenizer.downscale_multiplier)
    i_size_tokens = i_size // token_mult
    op_step = i_size_tokens // 2
    op_total = op_step * args.outpaint_range
    target_size = 2 * op_total + i_size_tokens
    mid = op_total + i_size_tokens//2
    for batch in islice(iter(dataset), args.example_count):
        initial_tokens = tokenizer.encode(batch)
        tokens = np.full([tf.shape(batch)[0].numpy(), target_size[0], target_size[1]], MASK_TOKEN, dtype=np.int32)
        tokens[:, op_total[0]:op_total[0]+i_size_tokens[0], op_total[1]:op_total[1]+i_size_tokens[1]] = initial_tokens
        total_steps = sum([4 * (s+2) for s in range(args.outpaint_range)])
        current_step = 0
        for ostep in range(args.outpaint_range):
            hs = (ostep + 1) * i_size_tokens // 2
            corners = np.array([mid-hs, mid+np.array([-hs[0], hs[1]]), mid+hs, mid+np.array([hs[0], -hs[1]])])
            side_steps = ostep + 2
            def get_pos(i):
                ci, si = divmod(i, side_steps)
                c1, c2 = corners[ci], corners[(ci+1)%4]
                return c1 + (c2 - c1) // side_steps * si 
            for sstep in range(4 * side_steps):
                #print ("#", flush=True)
                pred_pos = get_pos((sstep+1)%(4*side_steps))
                f, t = pred_pos-op_step, pred_pos+op_step
                new_tokens = maskgit.decode(tokens[:, f[0]:t[0], f[1]:t[1]])[0]
                tokens[:, f[0]:t[0], f[1]:t[1]] = np.where(tokens[:, f[0]:t[0], f[1]:t[1]] == MASK_TOKEN, new_tokens, tokens[:, f[0]:t[0], f[1]:t[1]])
                print (f"{current_step+1}/{total_steps} done.")
                current_step += 1

        final_img_size = [tf.shape(batch)[0], *(token_mult*target_size)]
        output_colors, overlaps = np.zeros([*final_img_size, 3]), np.zeros([*final_img_size, 1])
        ry, rx = np.arange(1, i_size+1), np.arange(1, i_size+1)
        blending = np.minimum(
            np.minimum(ry, ry[::-1])[:, np.newaxis],
            np.minimum(rx, rx[::-1])[np.newaxis, :]
        )[:, :, np.newaxis]
        for y in range(0, 1 + 2 * args.outpaint_range):
            for x in range(0, 1 + 2 * args.outpaint_range):
                f = np.array([y, x]) * op_step
                t = f+i_size_tokens
                colors = tokenizer.decode(tokens[:, f[0]:t[0], f[1]:t[1]])
                fi, ti = f * token_mult, t*token_mult
                output_colors[:, fi[0]:ti[0], fi[1]:ti[1]] += colors * blending
                overlaps[:, fi[0]:ti[0], fi[1]:ti[1]] += blending
        final_images = output_colors / overlaps
        wandb.log({"Images": [wandb.Image(i) for i in batch], "Outpainted":[wandb.Image(i) for i in final_images]})
        
                
                
                

    





if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)