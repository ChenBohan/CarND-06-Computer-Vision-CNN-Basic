# AI-CV-02-Intro-to-CNN
Udacity Self-Driving Car Engineer Nanodegree: Convolutional Neural Networks (CNN)

## Basic knowledge

Ref: https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/

Given:
- input layer has a width of ``W`` and a height of ``H``
- convolutional layer has a filter size ``F``
- a stride of ``S``
- a padding of ``P``
- the number of filters ``K``

The following formula gives us the width of the next layer: ``W_out =[(W−F+2P)/S] + 1``.

The output height would be ``H_out = [(H-F+2P)/S] + 1``.

And the output depth would be equal to the number of filters ``D_out = K``.

The output volume would be ``W_out * H_out * D_out``.

## Implement a CNN in TensorFlow

TensorFlow provides the ``tf.nn.conv2d()`` and ``tf.nn.bias_add()`` functions to create your own convolutional layers.

```python
# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```
The code above uses the ``tf.nn.conv2d()`` function to compute the convolution with ``weight`` as the filter and ``[1, 2, 2, 1]`` for the strides.

The ``tf.nn.bias_add()`` function adds a 1-d bias to the last dimension in a matrix.

## Pooling

News:

Recently, pooling layers have fallen out of favor. Some reasons are:

- Recent datasets are so big and complex we're more concerned about underfitting.
- Dropout is a much better regularizer.
- Pooling results in a loss of information. Think about the max pooling operation as an example. We only keep the largest of n numbers, thereby disregarding n-1 numbers completely.


### Max pooling

Max pooling operation is to reduce the size of the input, and allow the neural network to focus on only the most important elements.

- parameter-free --- Does not add to your number of parameters
- prevent overfitting
- often more accurate
- more expensive --- at lower stride
- more hyper parameters --- eg. pooling region size & pooling stride

TensorFlow provides the ``tf.nn.max_pool()`` function to apply max pooling to your convolutional layers.

```python
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

The ``ksize`` and ``strides`` parameters are structured as 4-element lists, with each element corresponding to a dimension of the input tensor (``[batch, height, width, channels]``). 

### Average pooling



## 1 * 1 convolutins

## Inception


