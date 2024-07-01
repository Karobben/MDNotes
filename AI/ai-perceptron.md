---
toc: true
url: ai_perceptron
covercopy: <a href="https://towardsdatascience.com/perceptrons-the-first-neural-network-model-8b3ee4513757">© Dr. Roi Yehoshua</a>
priority: 10000
date: 2024-02-07 13:03:23
title: "Perceptron"
ytitle: "Perceptron"
description: "Perceptron"
excerpt: "Perceptron"
tags: [Machine Learning, Data Science]
category: [Notes, Class, UIUC, AI]
cover: "https://miro.medium.com/v2/resize:fit:720/format:webp/1*gGmqkjA0VJCe5EhJnoQDNg.png"
thumbnail: "https://miro.medium.com/v2/resize:fit:720/format:webp/1*gGmqkjA0VJCe5EhJnoQDNg.png"
---

## Perceptron

Perceptron is invented before the loss function 


## Linear classifier: Definition

A linear classifier is defined by

$$
f(x) = \text{argmax } Wx + b
$$

$$ W \mathbf{x} + \mathbf{b} =  \begin{bmatrix} W_{1,1} & \cdots & W_{1,d} \\\\ \vdots & \ddots & \vdots \\\\ W_{v,1} & \cdots & W_{v,d} \end{bmatrix} \begin{bmatrix} x_1 \\\\ \vdots \\\\ x_d \end{bmatrix} + \begin{bmatrix} b_1 \\\\ \vdots \\\\ b_v \end{bmatrix} = \begin{bmatrix} \mathbf{w}_1^T \mathbf{x} + b_1 \\\\ \vdots \\\\ \mathbf{w}_v^T \mathbf{x} + b_v \end{bmatrix}
$$

where:

$w_k, b_k$ are the weight vector and bias corresponding to class $k$, and the argmax function finds the element of the vector $wx$ with the largest value.

|Multi-class linear classifier||
|:-|:-|
|![](https://imgur.com/TNWvhKX.png)|$ f(\mathbf{x}) = \arg\max (W\mathbf{x} + \mathbf{b}) $ <br>The boundary between class \( k \) and class \( l \) is the line (or plane, or hyperplane) given by the equation: <li>$ (\mathbf{w}_k - \mathbf{w}_l)^T \mathbf{x} + (b_k - b_l) = 0 $|






## Gradient descent

Suppose we have training tokens $(x_i, y_i)$, and we have some initial class vectors $w_1$ and $w_2$. We want to update them as

|||
|:-:|:-:|
|$w_1 \leftarrow w_1 - \eta \frac{\partial \mathcal{L}}{\partial w_1}$<br>$w_2 \leftarrow w_2 - \eta \frac{\partial \mathcal{L}}{\partial w_2}$ <br> ...where $\mathcal{L}$ is some loss function. What loss function makes sense?|![](https://imgur.com/YaSOBI6.png)




## Zero-one loss function

The most obvious loss function for a classifier is its classification error rate,

$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \ell(f(x_i), y_i)
$$

Where $\ell(\hat{y}, y)$ is the zero-one loss function,

$$
\ell(f(x), y) =
\begin{cases}
0 & \text{if } f(x) = y \\\\
1 & \text{if } f(x) \neq y
\end{cases}
$$

### Non-differentiable!

The problem with the zero-one loss function is that it’s not differentiable:

$$
\frac{\partial \ell (f(\mathbf{x}), y)}{\partial f(\mathbf{x})} = 
\begin{cases} 
0 & f(\mathbf{x}) \neq y \\\\
+\infty & f(\mathbf{x}) = y^+ \\\\
-\infty & f(\mathbf{x}) = y^- 
\end{cases}
$$




## One-hot vectors

One-hot vectors, A **one-hot vector** is a binary vector in which all elements are 0 except for a single element that’s equal to 1.

Take binary classification as an example:
  - class1: [1, 0]
  - class2: [0, 1]

The number of element in the list equals the number of classes.

Consider the classifier

$$
f(x) =  \begin{bmatrix} 
f_1(\mathbf{x}) \\\\
f_2(\mathbf{x}) 
\end{bmatrix} = \begin{bmatrix} 
\mathbb{1}_ {\arg\max W\mathbf{x}=1} \\\\
\mathbb{1}_ {\arg\max W\mathbf{x}=2} 
\end{bmatrix} 
$$



...where \(\mathbb{1}_P\) is called the "indicator function," and it means:

$$
\mathbb{1}_P = 
\begin{cases} 
1 & P \text{ is true} \\\\ 
0 & P \text{ is false} 
\end{cases}
$$

### Loss

## The perceptron loss

Instead of a one-zero loss, the perceptron uses a weird loss function that gives great results when differentiated. The perceptron loss function is:

$$
\ell(\mathbf{x}, \mathbf{y}) = (f(\mathbf{x}) - \mathbf{y})^T (W \mathbf{x} + \mathbf{b})
$$

$$
= [f_1(\mathbf{x}) - y_1, \cdots, f_v(\mathbf{x}) - y_v]  \begin{pmatrix} \begin{bmatrix}
W_{1,1} & \cdots & W_{1,d} \\\\
\vdots & \ddots & \vdots \\\\
W_{v,1} & \cdots & W_{v,d}
\end{bmatrix}
\begin{bmatrix}
x_1 \\\\
\vdots \\\\
x_d
\end{bmatrix}
+
\begin{bmatrix}
b_1 \\\\
\vdots \\\\
b_v
\end{bmatrix}
\end{pmatrix}
$$

$$
= \sum_{k=1}^{v} (f_k(\mathbf{x}) - y_k)(\mathbf{w}_k^T \mathbf{x} + b_k)
$$

## The perceptron learning algorithm


1. Compute the classifier output $\hat{y} = \arg\max_k (\mathbf{w}_k^T \mathbf{x} + b_k)$

2. Update the weight vectors as:

$$
\mathbf{w}_k \leftarrow \mathbf{w}_k - \eta \frac{\partial \ell(\mathbf{x}, \mathbf{y})}{\partial \mathbf{w}_k} = 
\begin{cases} 
\mathbf{w}_k - \eta \mathbf{x} & \text{if } k = \hat{y} \\\\
\mathbf{w}_k + \eta \mathbf{x} & \text{if } k = y \\\\
0 & \text{otherwise}
\end{cases}
$$

where $\eta \approx 0.01$ is the learning rate.

Because:

Because teh gradient of the perceptron loss is:

$$
\frac{\partial \ell(\mathbf{x}, \mathbf{y})}{\partial \mathbf{w}_k} = 
\begin{cases} 
\mathbf{x} & \text{if } k = \hat{y} \\\\
-\mathbf{x} & \text{if } k = y \\\\
0 & \text{otherwise}
\end{cases}
$$

So, we could have:

$$
\mathbf{w}_k \leftarrow 
\begin{cases} 
\mathbf{w}_k - \eta \mathbf{x} & k = \hat{y} \\\\
\mathbf{w}_k + \eta \mathbf{x} & k = y \\\\
0 & \text{otherwise}
\end{cases}
$$



<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
