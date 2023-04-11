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

The aforementioned dense layers have several problems for image data - the image has to be flattened to a large one-dimensional vector first, completely losing any spatial data, and using another dense layer on top requires an enormous amount of weights. To alleviate these issues, convolutional layers are used.

A convolutional layer can be viewed as an extension of the discrete convolution operation between two images, where one output channel is computed by 





Convolutional blocks, residual blocks, layer norm, ...


### Transformers



###