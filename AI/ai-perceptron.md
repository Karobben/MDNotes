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
There are a total of $v(d + 1)$ trainable parameters: the elements of the matrix $w$.

### Example

![](https://imgur.com/9kmw2fe.png)

Consider a two -class classification problem, with
-  $W^T_1 = [w_{1,1}, w_{1,2}] = [2,1]$
-  $W^T_2 = [w_{2,1}, w_{2,2}] = [1,2]$

Notice that in the two-class case, the equation

$$
f(x) = \text{argmax } Wx + b
$$

Simplifies to

$$
f(x) = 
\begin{cases} 
1 & \ if\ w_1^T x + b_1 > w_2^T x + b_2 \\\\
2 & \ if\ w_1^T x + b_1 \leq w_2^T x + b_2 
\end{cases}
$$


The class boundary is the line whose equation is
$$
(w_2 - w_1)^T x + (b_2 - b_1) = 0
$$

!!! note Extend: Multi-class linear classifier 

    ![](https://imgur.com/TNWvhKX.png)
    
    The boundary between class $k$ and class $l$ is the line (or plane, or hyperplane) given by the equation

|$f(x) = argmax Wx + b$| $(w_k - w_l)^T x + (b_k - b_l) = 0$|
|:-:|:-:|

The classification regions in a linear classifier are called Voronoi regions.
A **Voronoi region** is a region that is
• Convex (if $u$ and $v$ are points in the region, then every point on the line segment $\bar{u}\bar{v}$ connecting them is also in the region)
• Bounded by piece-wise linear boundaries


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

#### Exp2: Multi-Class

Consider the classifier

$$
f(x) = 
\begin{bmatrix}
f_1(x) \\\\
\vdots \\\\
f_v(x)
\end{bmatrix} =
\begin{bmatrix}
1_{\arg\max Wx=1} \\\\
\vdots \\\\
1_{\arg\max Wx=v}
\end{bmatrix}
$$

... with 20 classes. Then some of the classifications might look like this.

#### One-hot ground truth

We can also use one-hot vectors to describe the ground truth. Let’s call the one-hot vector $y$, and the integer label $y$, thus 

$$
y = \begin{bmatrix}
y_1 \\\\
y_2 \\\\ \end{bmatrix} = \begin{bmatrix}
1_{y=1} \\\\
2_{y=2} \end{bmatrix}
$$

Ground truth might differ from classifier output.

Instead of a one-zero loss, the perceptron uses a weird loss function that gives great results when differentiated. The perceptron loss function is:

$$
\ell(x, y) = (f(x) - y)^T (Wx + b)
$$

$$
= \left[ f_1(x) - y_1, \ldots, f_v(x) - y_v \right]
\left(\begin{bmatrix}
W_{1,1} & \ldots & W_{1,d} \\\\
\vdots & \ddots & \vdots \\\\
W_{v,1} & \ldots & W_{v,d}
\end{bmatrix}
\begin{bmatrix}
x_{1} \\\\
\vdots \\\\
x_{d}
\end{bmatrix}
+
\begin{bmatrix}
b_{1} \\\\
\vdots \\\\
b_{v}
\end{bmatrix}\right)
$$

$$
= \sum_{k=1}^{v} (f_k(x) - y_k)(W_k^T x + b_k)
$$


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

### Special case: two classes

If there are only two classes, then we only need to learn one weight vector, $w = w_1 - w_2$. We can learn it as:

1. Compute the classifier output $\hat{y} = \arg\max_k (w_k^T x + b_k)$

2. Update the weight vectors as:

$$
w \leftarrow 
\begin{cases} 
w - \eta x & \text{if } \hat{y} \neq y, y = 2 \\\\
w + \eta x & \text{if } \hat{y} \neq y, y = 1 \\\\
w & \text{if } \hat{y} = y
\end{cases}
$$

where $\eta \approx 0.01$ is the learning rate. Sometimes we say $y \in \{1, -1\}$ instead of $y \in \{1,2\}$.





<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
