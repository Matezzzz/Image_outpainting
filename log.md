#13.3. 

Working on basic image segmentation so I can use the webcam data (I want those because there is much more of it, I could train a model from scratch just for clouds)

* Based on color doesn't work very well - bad at night or when the sky color changes 
* Idea - areas that should be thresholded will vary in color a lot locally (as opposed to sky, that will be homogenous most of the time)
    * Approximate color based on the surroundings -> measure error -> places that often have high error are not a part of the sky
        * Tried using a simple NN as approximation - it had problem with small values, and was kinda useless :D
        * Just use linear averaging from the surrounding pixels
            * A bit noisy, but works for images during daytime
            * add closing, fill small empty regions, empty small filled regions, if there is a region spanning from left to right border, fill everything below it
        * If a mask occurs at multiple consecutive times, it will be in the final mask too


#14.3.
