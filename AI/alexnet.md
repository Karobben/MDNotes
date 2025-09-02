---
toc: true
url: alexnet
covercopy: <a href="https://wikidocs.net/165426">© 고민수</a>
priority: 10000
date: 2025-02-06 15:39:05
title: "AlexNet"
ytitle: "AlexNet"
description: "AlexNet is a convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge in 2012."
excerpt: "AlexNet is a convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge in 2012. It was designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. The network has eight layers, five of which are convolutional layers and three are fully connected layers. It uses ReLU activation functions, dropout for regularization, and data augmentation techniques to improve performance. AlexNet significantly advanced the field of deep learning and computer vision."
tags: [neuralnetworks, deeplearning, computer-vision]
category: [AI, Machine Learning]
cover: "https://wikidocs.net/images/page/164787/AlexNet-Fig_03.png"
thumbnail: "https://wikidocs.net/images/page/164787/AlexNet-Fig_03.png"
---

## AlexNet

AlexNet is a convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge in 2012. It was designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. The network has eight layers, five of which are convolutional layers and three are fully connected layers. It uses ReLU activation functions, dropout for regularization, and data augmentation techniques to improve performance. AlexNet significantly advanced the field of deep learning and computer vision.

In the structure illustration, it was usually split into 2 parts. It is because backing that time, GPU has limited memory. So, they split each layer into 2 parts, and each part is trained on a different GPU.

![AlexNet, wikipedia](https://upload.wikimedia.org/wikipedia/commons/c/cc/Comparison_image_neural_networks.svg)

As you can see from this illustration, the input imge is 227x227x3, which is then passed through five convolutional layers, followed by three fully connected layers. The output is a 1000-dimensional vector representing the probabilities of the image belonging to each of the 1000 classes in the ImageNet dataset.

Let's break down the architecture of AlexNet:
1. **Input Layer**: The input layer takes an image of size 227x227x3 (height, width, channels or RGB values).
2. **Convolutional Layer 1**: The first convolutional layer applies 96 filters of size 11x11 with a stride of 4, resulting in an output of size 55x55x96. This layer uses ReLU activation and is followed by a max pooling layer with a 3x3 window and a stride of 2.
    - What does ReLU do? It introduces non-linearity into the model by replacing all negative pixel values in the feature map with zero. By doing this, the model can learm more complex patterns in the data. So, it basically makes the negative values as a closed signal. 
3. **Convolutional Layer 2**: The second convolutional layer applies 256 filters of size 5x5 with a stride of 1, resulting in an output of size 27x27x256. This layer also uses ReLU activation and is followed by a max pooling layer with a 3x3 window and a stride of
...
4. Finally, we got a 5x5x256 feature map. This is then flattened into a 1D vector of size 6400. It then passes through three fully connected layers with 4096, 4096, and 1000 neurons, respectively. The final layer uses a softmax activation function to output the probabilities of the image belonging to each of the 1000 classes in the ImageNet dataset.


So, as you can see, the original image contains 227x227x3 = 154,587. In daily life, the picture we using are 1920*1080, which could have 6,220,800 input. With this convolutional techniques, we can reduce the size of the image while still keeping the important information.


!!! note **Terminology** Explained
    - **filters**: The filters are also known as kernels, and they are used to detect features in the input image. In this case, the first convolutional layer uses 96 filters of size 11x11.
    - **stride**: The stride is the number of pixels by which the filter is moved across the image. You can also think of it as the step size. In this case, the stride is 4, which means that the filter is moved 4 pixels at a time. **55** = (227 - 11) / 4 + 1
    - **5.6 times of decreasing**:  Now, instsad of 227x227x3, we have 55x55x96. You can image that the image is now much smaller, but it still contains a lot of information. Though, the "pixels" are less than the original image, each information from the single pixel is now represented by 96 values rather than 3 values. By multiplying the number of pixels with the number of filters, we can see that the amount of information has increased significantly. (227x227x3 = 154,587, 55x55x9 = 27225)
    - **ReLU activation**: The ReLU activation function is applied to introduce non-linearity into the model. It replaces all negative pixel values in the feature map with zero.
    - **Max Pooling**: The max pooling layer reduces the spatial dimensions of the feature map by taking the **maximum value in each 3x3 window** with a stride of 2. This helps to reduce the computational complexity and makes the model more robust to variations in the input image. The difference between the convolutional layer and the max pooling layer is that the convolutional layer applies a filter to the input image, while the max pooling layer reduces the spatial dimensions of the feature map by taking the maximum value in each window.

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
