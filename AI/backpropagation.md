---
toc: true
url: backpropagation
covercopy: © Karobben
priority: 10000
date: 2025-04-28 12:53:29
title: "Backpropagation Hand by Hand"
ytitle: "Backpropagation Hand by Hand"
description: "Backpropagation Hand by Hand"
excerpt: "Backpropagation is the algorithm that trains neural networks by adjusting their weights to minimize the loss. It works by applying the chain rule of calculus to efficiently compute how the loss changes with respect to each weight. Starting from the output layer, it propagates the error backward through the network, layer by layer, updating the weights based on their contribution to the error.$$W^\\ell\\!\\leftarrow W^\\ell - \\eta\\, (a^ {\\ell-1}) ^\\top \\delta^\\ell,\\quad b^\\ell\\!\\leftarrow b^\\ell - \\eta\\,\\sum\\delta^\\ell.$$"
tags: [Machine Learning, Data Science]
category: [Machine Learning]
cover: ""
thumbnail: ""
---

$$
    W^\ell\!\leftarrow W^\ell - \eta\, (a^ {\ell-1}) ^\top \delta^\ell,\quad
    b^\ell\!\leftarrow b^\ell - \eta\,\sum\delta^\ell.
$$

## A Very Sample NN Example


Codes below is a simple example of a neural network with one hidden layer, using the sigmoid activation function. The network is trained to learn the XOR function, which is a classic problem in neural networks. The **Input** consists of two features. The **Output** is a single value (either 0 or 1). The network has two layers: an **input layer** with 2 neurons and a **hidden layer** also with 2 neurons. The **output layer** has 1 neuron. The **weights** and **biases** are *initialized randomly*. The network is trained using the **backpropagation** algorithm, which adjusts the weights and biases based on the error between the predicted output and the actual output.



```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Example input and output
# There are 4 samples, and each sample has 2 features (e.g., like a point in 2D space).
x = np.array([[0, 0], [0, 1], [1, 0], [-1, 1]])
# Each sample has a single output (either 0 or 1).
y = np.array([[0], [1], [1], [0]])

#########################################
# Step 1. Initialize weights and biases
#########################################
W1 = np.random.rand(2, 2)  # (input_dim=2 → hidden_dim=2)
b1 = np.random.rand(1, 2)  # bias for hidden layer (1 row, 2 neurons)
W2 = np.random.rand(2, 1)  # (hidden_dim=2 → output_dim=1)
b2 = np.random.rand(1, 1)  # bias for output layer (1 row, 1 neuron)

# Learning rate
eta = 0.1

# Training loop
for epoch in range(10000):
    ############################################
    # Step 2. Forward pass
    ############################################
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    ############################################
    # Step 3. Loss calculation
    ############################################
    loss = 0.5 * (y - a2)**2
    ############################################
    # Step 4. Backpropagation
    ############################################
    delta2 = (a2 - y) * sigmoid_derivative(a2)
    delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(a1)
    # Update weights and biases
    W2 -= eta * np.dot(a1.T, delta2)
    b2 -= eta * np.sum(delta2, axis=0, keepdims=True)
    W1 -= eta * np.dot(x.T, delta1)
    b1 -= eta * np.sum(delta1, axis=0, keepdims=True)

print("Final output after training:")
print(a2)
```

## Steps of Backpropagation

1. **Initialization:** (Step 1)
   - Initialize the weights and biases of the network with small random values.

2. **Forward Pass:** (See Step 2 in the for loop)
   - For each layer $ l $, compute the input $ z^l $ and output $ a^l $:
     - $z^l = W^l a^{l-1} + b^l$
     - $a^l = \sigma(z^l)$
   - Here, $ W^l $ are the weights (`W1` and `W2`), $ b^l $ are the biases (`b1` and `b2`), $ \sigma $ is the activation function (`sigmoid`), and $ a^{l-1} $ is the output from the previous layer (the first $a$ is $x$ which is the input).

3. **Compute Loss:** (See codes in Step 3)
   - Compute the loss $ L $ using a suitable loss function.

4. **Backward Pass:** (See codes in Step 4)
   - Calculate the gradient of the loss with respect to the output of the last layer $ \delta^L $:
        - $\delta^L = \nabla_a L \cdot \sigma'(z^L)$
   - For each layer $ l $ from $ L-1 $ to 1, compute:
        -$\delta^l = (\delta^{l+1} \cdot W^{l+1}) \cdot \sigma'(z^l)$
   - Update the weights and biases:
        - $W^l = W^l - \eta \cdot \delta^l \cdot (a^{l-1}) ^T$
        - $b^l = b^l - \eta \cdot \delta^l$
   - Here, $ \eta $ is the learning rate, and $ \sigma' $ is the derivative of the activation function.


$$
\frac{d(\text{Loss})}{dW_2} = \frac{d(\text{Loss})}{da_2} \times \frac{da_2}{dz_2} \times \frac{dz_2}{dW_2}
$$

- $ \frac{d(\text{Loss})}{da_2} $: how much the loss changes if $ a_2 $ changes
- $ \frac{da_2}{dz_2} $: how much $ a_2 $ changes if $ z_2 $ changes
- $ \frac{dz_2}{dW_2} $: how much $ z_2 $ changes if $ W_2 $ changes



<details><summary>Click to go through this step by step very carefully</summary>


The delta calculation in the code:
```python
delta2 = (a2 - y) * sigmoid_derivative(a2)
```

and how we get it based on chain rule:
$$
\frac{d(\text{Loss})}{dW_2} = \frac{d(\text{Loss})}{da_2} \times \frac{da_2}{dz_2} \times \frac{dz_2}{dW_2}
$$


**Step 1: Loss function**

Suppose the loss is Mean Squared Error (MSE):

$$
\text{Loss} = \frac{1}{2} (a_2 - y)^2
$$

**Step 2: Derivatives**

| Main   | Derive from    | Result|
|---------------- | --------------- | --------------- |
| $ \frac{d(\text{Loss})}{da_2} $    |  $Loss = 0.5 * (y - a_2)^2 $   |   $(a_2 - y)$  |
| $ \frac{da_2}{dz_2} $    | $a_2 = \sigma(z_2)$ | $\sigma'(z_2)$    |
| $ \frac{dz_2}{dW_2} $    | $ z_2 = a_1 W_2 + b_2 $ |$\frac{dz_2}{dW_2} = a_1$| 

$\sigma'(z_2) = a_2 (1 - a_2)$

---

**Step 3: Chain them together**

Thus:

$$
\text{delta2} = \frac{d(\text{Loss})}{da_2} \times \frac{da_2}{dz_2}
$$
which is exactly:
```python
delta2 = (a2 - y) * sigmoid_derivative(a2)
```

Later, the update for $ W_2 $ uses $ delta2 $ multiplied by $ a_1 $ (previous activations).

---

**Summary:**
- `(a2 - y)` is $\frac{d(\text{Loss})}{da_2}$
- `sigmoid_derivative(a2)` is $\frac{da_2}{dz_2}$
- We haven't yet multiplied by $a_1$ — that happens during weight update!

</details>

!!! question Why we made delta2 vector without multiplying by $a_1$?
        Except we will use it to update $W_2$ later by multiplication with $a_1$, we also need to update **$b_2$** which doesn't require multiplication $a_1$. So, we don't need to multiply $a_1$ here.


## Expansions  

1. **Derive δ¹ Explicitly**  
   Work through the chain rule for the hidden layer:  
   $$
     \delta^1 = \bigl(\delta^2 W^{2\top}\bigr)\odot \sigma'(z^1).
   $$  
   Mapping each term to code deepens your grasp of how errors “flow back” through layers  ([Backpropagation calculus - 3Blue1Brown](https://www.3blue1brown.com/lessons/backpropagation-calculus?utm_source=chatgpt.com)).

2. **Bias Gradients**  
   Note that  
   $\partial L/\partial b^\ell = \sum_i \delta^\ell_i$,  
   which the code implements via `np.sum(deltaℓ, axis=0, keepdims=True)`  ([Backpropagation - Wikipedia](https://en.wikipedia.org/wiki/Backpropagation?utm_source=chatgpt.com)).

3. **Alternative Losses**  
   Show how the backprop equations simplify when using binary cross‐entropy:  
   $$
     L = -y\log(a^2) - (1-y)\log(1-a^2)
     \quad\Longrightarrow\quad
     \delta^2 = a^2 - y,
   $$  
   eliminating the explicit $\sigma'(z^2)$ term  ([Backpropagation - Wikipedia](https://en.wikipedia.org/wiki/Backpropagation?utm_source=chatgpt.com)).

4. **Regularization in Loss**  
   Introduce weight decay by adding $\tfrac{\lambda}{2}\|W\|^2$ to $L$.  Its gradient $\lambda W$ simply augments $\partial L/\partial W$ before the update  ([History of artificial neural networks](https://en.wikipedia.org/wiki/History_of_artificial_neural_networks?utm_source=chatgpt.com)).

5. **Vectorizing Over Batches**  
   Generalize from a single example to a batch by keeping matrices `X` and `Y` shaped `(batch_size, features)`, ensuring the update formulas remain the same  ([Backpropagation - CS231n Deep Learning for Computer Vision](https://cs231n.github.io/optimization-2/?utm_source=chatgpt.com)).

6. **Modern Optimizers**  
   Briefly demonstrate how to plug in Adam: maintain per-parameter running averages `m` and `v` and adjust `W` with  
   $\displaystyle m\leftarrow\beta_1 m + (1-\beta_1)\nabla W,\quad
    v\leftarrow\beta_2 v + (1-\beta_2)(\nabla W)^2$,  
   then  
   $$
     W\leftarrow W - \eta\,\frac{m/(1-\beta_1^ t)}{\sqrt{v/(1-\beta_2^ t)}+\epsilon}.
   $$  
   This typically outperforms vanilla SGD on noisy or sparse data  ([History of artificial neural networks](https://en.wikipedia.org/wiki/History_of_artificial_neural_networks?utm_source=chatgpt.com)).














<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>

