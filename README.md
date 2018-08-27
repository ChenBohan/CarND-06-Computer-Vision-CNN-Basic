# AI-CV-02-Intro-to-CNN
Udacity Self-Driving Car Engineer Nanodegree: Convolutional Neural Networks (CNN)

## Basic knowledge

The best way to explain a conv layer is to imagine a flashlight that is shining over the top left of the image.

- ``filter``: This flashlight is called a filter(or sometimes referred to as a neuron or a kernel).
- ``receptive field``: The region that it is shining over.


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
