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