# Cloud image outpainting using masked generative transformers


# Abstract

This work attempts to create a program that is able to expand an image of the sky and expand it beyond it's borders using neural networks. We use state-of-the-art techniques to produce high quality imagery trained only on cloud data.




# Background

In this chapter we familiarize the reader with several topics from machine learning that are used in this thesis. This includes many types of neural networks such as convolutional networks, transformers, and their methods of training.



## Neural networks

Neural networks are a type of machine learning model, composed of multiple layers (each of which is parametrized by some weights) and activation functions, each one represented by a mathematical operation. Each layer either operates on the output from a previous layer or on the model input, making the whole model a composite function, producing an output given input and the model weights.

The model trains by tweaking the weights of it's layers in order to minimize a loss function, which measures how well the model performs on a given task. 

We first introduce a basic layer and an activation function, on which we show the basics of how a model learns. After that, we show more advanced layers that will be later used to outpaint 

### Introduction

The simplest neural network one can build is called a multi layer perceptron. It consists of an input $I \in R^{n}$, one dense hidden layer, and one dense output layer. A dense layer has a matrix of weights $W \in R^{m \cdot n}$, where $m$/$n$ are the input/output dimensions respectively, and a specified activation function $\sigma$. The layer output $y$ is then defined as $y = \sigma(x^T \cdot W)$, where $\sigma$ is applied to each element of the output vector individually.

[DENSE NET IMAGE]

When we provide the input value $I$, we can compute the value of the first dense layer using it's formula, then we repeat the same for the second dense layer, producing the network output.

### Training

In order to train the network, we need a training set containing examples that can be passed as input to the first layer and their respective outputs that define what the model should return for the given value. We also need to provide a loss function $L$, which serves as a measure of how far our model outputs are from the correct ones, defining what value should our model minimize. Commonly used loss functions include mean squared error for regression tasks and sparse categorical crossentropy for classification.

Neural networks are trained using gradient descent - we want to compute the derivative of the loss function with respect to each weight, and then update each weight using the following formula: 

$$w' = w + \alpha \cdot \frac{\partial L}{\partial w}$$

We begin by passing the example to the network input, evaluating the outputs and computing the loss function. After that, we compute the derivatives of all weights with respect to the loss using the *back-propagation* algorithm [REF].

In practice, there are two major distinctions - the gradient is averaged over multiple examples, called a batch, in order to produce a more informed estimate, and the weights are updated using an optimizer - an algorithm that is provided with the gradient, can modify it in some way, and only then updates the weights. The optimizers we use in this work are a variation of Adam [REF], which also tracks the momentum of the gradient over multiple batches, and takes this information into account when doing an update.


### Convolutional networks

#### Convolution basics

The aforementioned dense layers have several problems for image data - the image has to be flattened to a large one-dimensional vector first, completely losing any spatial data, and using another dense layer on top requires an enormous amount of weights. To alleviate these issues, convolutional layers were introduced in [REF].

A convolutional layer can be viewed as an extension of the discrete convolution operation between two images. When using one input and one output layer, the output at a given position is computed by multiplying the surrounding pixels in the input by a convolutional matrix and applying an activation to the result, the values of which are the same for every position in the output and which represent the trainable weights. When using multiple input channels, each layer in the input has a separate convolution matrix, and the convolution results are summed before applying the activation. When using multiple outputs, each additional layer is computed in the same way as the first, just using a different matrix of weights.

Convolutional layers are great at working with data locally, but unless an extensive amount of layers is used, they have no way to propagate information to the other side of the image. For this reason, downscaling layers were introduced - by using a greater stride between individual matrix placements in the input, they are able to downscale the image, compressing the information to a smaller size. When upscaling is required instead, a transposed convolution, first introduced in [REF], can be used.



#### Residual connections & normalization

Convolutional networks require many layers, each of which performs a small part of the whole computation. For large networks, possible issues like *vanishing/exploding gradients* [REF] and increasingly more complex loss landscapes [REF] become more prominent. To adress this, ResNet [REF] proposes residual connections, which boil down to taking an input, applying a series of layers, and then adding the result to the original input, forming a residual block. In the context of convolutional networks, a single block may look as follows:

[RESIDUAL CONVOLUTIONAL BLOCK IMAGE]

Another thing helping with training are new forms of regularization in the form of normalization layers, which all take some input data and try to shift and scale it so that it has a variance of 1 and a mean of 0. The first to be proposed was a batch normalization layer [REF], tracking the mean and variance of each layer in a convolutional network, and shifting the values in each layer to conform to the properties mentioned above. More advanced forms of normalization appeared later, such as layer normalization [REF] and group normalization [REF], averaging values in each example individually and over groups of layers in an example respectively.


### Transformers

Originally introduced in the field of natural language processing [REF] as a successor to recurrent networks, transformers allow working with sequences of elements, where any element can be affected by all others. A typical transformer layer consists of two parts, a multi-head attention, which is responsible for interactions between multiple elements, and a dense layer processing part, which processes each sequence element individually.


....

### VQVAE?


### MaskGIT


### Diffusion models




## The Algorithm

This chapter describes all models used, how they are trained, and how they are utilized in the outpainting algorithm.

Our outpainting algorithm consists of three distinct models - a VQVAE tokenizer, the MaskGIT transformer model, and a diffusion model upscaler. The tokenizer consists of two parts, an encoder that is able to convert images to a latent space consisting of discrete tokens, and a decoder, which attempts to reconstruct the original image given tokens. The MaskGIT model takes in an image converted to tokens using the tokenizer, with some tokens getting masked out, and tries to reconstruct the original tokens. The upscaler gets an image as input, upscales it, and tries to fill in the missing details.

During outpainting, we use the models as follows - first, we get the image we are trying to extend, pass it through the tokenizer encoder, receiving a set of tokens. After that, we repeatedly call the MaskGIT model, generating feasible tokens surrounding the original image, until a specified size is reached. When we are done, we use the tokenizer decoder to convert the tokens back to an image, which is then upscaled using the upscaler to create the final result.

We will start by describing the dataset used to train the models. After that, we will show how every model is trained, and finally, we will show the details of how they work together during the outpainting process.

[IMAGE DETAILING THE PROCESS?]

### Dataset

For training the models, we use the data graciously provided by the czech hydro-meteorological institute (further referred to as CHMI) [REF?] and collected by the computer graphics group [REF?] at Charles university. The data consists of 98 locations, each of which has a static webcam that saves an image of resolution 1600x1200 every 5 minutes. We use all the data available, which was being collected from March in year 2021 to the time of writing, summing up to circa 200 thousand images per location, or approximately 19 million in total.

All images contain some part of the landscape, the CHMI logo in the left top corner, and some weather measurements in the top right, in addition to the sky. For this reason, we start by performing a simple segmentation on each location, figuring out which pixels are part of the sky and which ones are not, assuming that these do not change over the time of collecting the data. These masks are generated once before training, and their creation is described in more detail in the section below. When preparing the dataset for training a particular model, we downscale the images and masks and  select a random part of the sky as model input, then filter the selected parts to condition the model to create more diverse images. In the following subsections, we first describe the segmentation algorithm in detail, then we describe how it is used to generate training data for each model.


#### Location segmentation

We base location segmentation on the following observation - during a day, the sky will change quite a lot, while the landscape stays mostly the same. The  algorithm then boils down to detecting edges in an image, doing some processing, and then finding the places where edges are present many times during different times of a day. We describe the process of creating a segmentation mask, in which unmasked pixels are part of the sky, and masked ones are part of the landscape, the CHMI logo, or the weather information text.

I detect edges by computing the image gradient with respect to the x and y axis and summing the absolute values of both directions and all rgb components. After that, we use a threshold, marking all values above it as parts of the landscape.

I then process the edge mask as follows - first, we perform the morphological closing operation [REF] with a 3x3 square mask five times in a row, making parts of the mask a lot more cohesive and filling in noisy areas. After that, we find all continuous non-landscape areas in the mask and mark all areas whose number of pixels is below a certain threshold (20000 pixels is used) as landscape. Then, we do the same for areas marked as landscape, discarding those with less than 300 pixels, considering them random noise.

After this, we hide the CHMI logo by masking a 270x115 rectangle in the top right and the weather measurements by masking a 250x250 rectangle in the top left.

[SEGMENTATION RESULTS?]

During daytime with bright skies, this generally produces an acceptable mask, with just few mistakes, such as masked parts of clouds and missing masks over patches of landscape of the same color. To fix these issues, we compute separate masks for a hundred consecutive time steps from one day (starting at noon so most time steps are during the day). Then, for each pixel, we compute the largest consecutive amount of time steps it has been marked as an edge, and the percentage of the total time it was an edge. In order for an edge to be in the final mask, the consecutive amount must be above a certain threshold, and the percentage must be above another threshold.

We didn't manage to find thresholds that would work well for all locations and weather conditions, so we instead chose the thresholds manually for all locations. The final segmentation masks look as follows:

[FINAL MASKS]


#### Generating training data

We now describe how to generate a training example, assuming we have a loaded image and its' segmentation mask, both of size 1600x1200. When mentioning color constants in the following paragraphs, we assume all images to be normalized with values between 0 and 1.

We generate training data by downscaling both the image and the mask by some factor, and selecting a random subset that contains only sky pixels according to the provided mask. We believe this is better than simply scaling the image as it enables us to eliminate unwanted stretching artifacts and having to worry about masking landscape pixels during training. It also conditions the networks to learn exclusively about the sky, which is what we want to generate during outpainting. The tokenizer and MaskGIT networks take inputs of size 128x128, so we found it useful to downscale the images by a factor of 4 before selecting the area, allowing the training to select a large part of the sky while still fitting multiple positions above the landscape. As the upscaler takes images of size 512x512, we don't downscale at all.

We also found it useful to filter the images created by the procedure above in order to condition networks to generate more interesting images. First, we discard all images that were recorded during the night - because the downscaled training data doesn't have a high enough resolution to capture any stars or planets on the night sky, all the images are pitch black, and there is nothing interesting to generate. We consider all images with mean less than 0.2 to be those of the night sky, and discard all of them, filtering out about 40% of all input images.

We also try to increase variability in the generated images by limiting the amount of monochrome images present in the training data. We found that using the unrestricted dataset makes it highly likely that we will get a single-color image during outpainting as well. Because we want to drastically limit this behaviour, but not eliminate generating monochrome images altogether, we choose to keep 10% of the monochrome images with standard deviation from their mean less than 0.05, and all others, filtering out another 50% of the dataset. 

Although we eliminate quite a large chunk of the training data during this step, we find it worth the increased quality in generated samples, and no measurable decrease in performance, partly due to the abundance of data available.

[FILTERING EXAMPLE??]



### The models

In this section, we describe all the models that will be used in the outpainting algorithm, how they are trained, and their results.

#### Tokenizer

The tokenizer is responsible for taking in images and converting them to a latent space consisting of discrete tokens. The architecture is based on the vector quantization variational autoencoder [REF] and consists of three parts - an encoder, the vector quantization layer, and a decoder. The encoder takes images as an input, processing them using multiple residual blocks and downscaling convolutions, and outputs a downscaled image of embedding vectors in an embedding space. The vector quantization layer then takes the vectors in embedding space, and attempts to represent each of them using a discrete token, then it converts the tokens back to the embedding space. The decoder then uses more residual layers and transposed convolutions to attempt to reconstruct the original image from the vectors in the embedding space. We describe the architecture and training process of all model parts in this section.



The encoder 



#### MaskGIT




#### Diffusion model upscaler




!!! tokenizer linear output + squish

