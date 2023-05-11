# Generative neural networks for sky image outpainting

The files in this directory are a part of my thesis about sky image outpainting at Charles University in Prague. Here, we present a short guide describing how to run the project - this is a regurgitation of the summary available in the thesis (available at *thesis/thesis.pdf* at [GitHub](https://github.com/Matezzzz/Image_outpainting)) with less details, but for completion, we provide it anyway.

## Quick start

The project requires Python 3.10, Tensorflow 2.11 and several other libraries. Download python, then install tensorflow according to the [official guide](https://www.tensorflow.org/install). When done, install all other libraries using the command `pip install numpy tensorflow-probability Pillow wandb python-opencv scikit-image paramiko scipy`.

Create a folder in the same directory this readme is in and name it `models`.

Continue by downloading all three models from [google drive](https://drive.google.com/drive/folders/10531XM-ujFrR6iDo_SHgLLb03tbFEShG?usp=sharing), or use the ones downloaded alongside the thesis attachments. After it is done, extract each `.zip`, and place them into the `./models` folder. Be wary of the directory structure - the `saved_model.pb` should be present in the directory of each model (e.g. for the tokenizer, `models/tokenizer_228/saved_model.pb` should exist).

Place all images you want to outpaint in the `./outpaint` folder, then run the `outpainting.py` script. Each image will be resized to $128 \times 128$, then it will be outpainted. Results can be viewed online using the weights and biases library or in the `./outpaint_results` folder.

Run `python outpainting.py --help` to display all available parameters, feel free to tweak them and run outpainting again.


## Browsing source code

We provide a short decription of each file and folder in the project, to get started with browsing:

* `tokenizer.py` For training the VQVAE tokenizer
* `maskgit.py` For training the MaskGIT model. A trained tokenizer is required to run
* `diffusion_model_upscale.py` For training the super sampler. A trained tokenizer is required to run
* `outpainting.py` For running outpainting
* `build_network.py` Utilities to make building networks in tensorflow simpler
* `dataset.py` Image dataset loading, filtering, and segmentation according to known masks
* `log_and_save.py` Logging training metrics to wandb, useful callbacks
* `tf_utilities.py` Function for initializing tensorflow
* `utilities.py` Multiple functions for working with files, getting filenames and more
* `check_dataset.py` Go over all images in a dataset and notify the user about those with broken formatting
* `clean_up_wandb.py` Delete old model versions from weights and biases
* `segmentation.py` Create segmentation masks
* `./masks` Contains masks for all locations. During training, only the locations from here will be used.
* `./data` During the first run, we search for all training data matching
`(dataset_location)/place/*/*.jpg`, where place has an available mask at `./masks/(place)_mask.png`. This creates two files, `./data/masks.npy.gz`, and `masks/filename_dataset.data`, which will be used in all subsequent runs
* `./models` All models will be loaded from or saved to this directory
* `./outpaint` Stores images to outpaint


## Training models

To train a model, a dataset is required. We use photos from the [CHMI website](https://www.chmi.cz/files/portal/docs/meteo/kam/) collected over several years. This dataset is not provided as part of the project. A custom dataset can be scraped from the website if required. We describe shortly how to prepare the data:
 * We only support czech locations fully, for these, use the masks provided with the project. For other places, use the `segmentation.py` file - it contains two methods, `image_segmentation` that computes a mask and saves it, and `finalize_masks` that allows tweaking the created mask. Refer to the source code for more details. All masks created in this way will be saved into `./masks` automatically.
 * Mind that the first run will go through all the files in the data directory and save them, which can take quite a lot of time based on the amount of files present
 * (Optional) Verify that data has been loaded without a problem by running the `dataset.py` method. It will log examples on which we later train to the weights and biases server.
 * Train a model by running any training script. Additional parameters tweaking the model behaviour are supported. Training progression can be viewed using weights and biases.
