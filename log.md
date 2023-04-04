#12.3. 

Working on basic image segmentation so I can use the webcam data (I want those because there is much more of it, I could train a model from scratch just for clouds)

* Based on color doesn't work very well - bad at night or when the sky color changes 
* Idea - areas that should be thresholded will vary in color a lot locally (as opposed to sky, that will be homogenous most of the time)
    * Approximate color based on the surroundings -> measure error -> places that often have high error are not a part of the sky
        * Tried using a simple NN as approximation - it had problem with small values, and was kinda useless :D
        * Just use linear averaging from the surrounding pixels
            * A bit noisy, but works for images during daytime
            * add closing, fill small empty regions, empty small filled regions, if there is a region spanning from left to right border, fill everything below it
        * If a mask occurs at multiple consecutive times, it will be in the final mask too

#13.3.

* Should implement the neural nets - I will rewrite the provided implementation in tensorflow, just read the code so far

#14.3.

* start implementing the tokenizer
    * Using a slightly different upscaling in the decoder (residual instead of nearest scale + convolution)
    * Some operations are slightly different compared to original maskgit so the code is simpler (entropy loss is missing for example)

#15.3

* tokenizer doesn't train, latent losses suspiciouslty low. Try training without quantizer + codebook -> was due to a bug. Now just a VAE works
* codebook works as well. Samples are quite low quality and latent losses are still extremely small
* Problem - ordered images, need a larger shuffle limit
* Ditched subclassing for functional API - we will use slightly more memory when training, but saving should be easier

#16.3.
* training works! I added a loss that moves codebook keys to the closest vector - this should avoid completely unused vectors


#24.3.

*maskgit training
 * 93 - no transformer, just dense -> kinda clueless, but can copy patches
 * 95 - no transformer + sample_weights
 * 96 - one transformer + sample weights
 * 98 - all transformers, residual connections
 * 101 - one transformer, maskgit logits
 * 102 - two transformers, maskgit logits
 * 107 - one transformer, actual decoding, maskgit logits
 * 109 - two transformers, actual decoding, maskgit logits, smaller batch size (16 out of 32)


#27.3.

* v2 ideas
    * discriminatory loss for tokenizer (+noise?)
    * half image training for maskgit
    * try diffusion models?

#28.3.
 * started using batch norm instead of group norm to lower memory consumption (and went back because )
 

#29.3.
 * sparse top k accuracy is too high -> suspicious. Might have only a few vectors that are used a lot! Would explain the slow tokenizer training
 * revert to group norm



#2.4.

trying to improve images - add details
 * diffusion models - is good at generation, but hard to use for adding details


#3.4.
 * Try going back to sparse CCE (now using NLL)

 * try discrete image space - will be useless, but might be fun


 * Fix maskgit (validation error, whoops)!!!
