# Cloud image outpainting using masked generative transformers


# Abstract

This work attempts to create a program that is able to expand an image of the sky and expand it beyond it's borders using neural networks. I use state-of-the-art techniques to produce high quality imagery trained only on cloud data.




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






###