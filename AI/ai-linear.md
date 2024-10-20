---
toc: true
url: ai_linear
covercopy: <a href="https://www.geeksforgeeks.org/ml-linear-regression/">© geeksforgeeks</a>
priority: 10000
date: 2024-02-05 12:26:13
title: "Linear Regression"
ytitle: "Linear Regression"
description: "Linear Regression"
excerpt: "Linear Regression"
tags: [Machine Learning, Data Science]
category: [Notes, Class, UIUC, AI]
cover: "https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png"
thumbnail: "https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png"
---

## Linear Regression

### Vectors and Matrix

In numpy, the dot product can be written np.dot(w,x) or w@x.
Vectors will always be column vectors. Thus:

$$
x = 
\begin{bmatrix}
x_{1} \\\\
\vdots \\\\
x_{n}
\end{bmatrix}
, \quad w^T = [w_{1}, \ldots, w_{n}]
$$

$$
w^Tx = [w_{1}, \ldots, w_{n}]
\begin{bmatrix}
x_{1} \\\\
\vdots \\\\
x_{n}
\end{bmatrix}
= \sum_{i=1}^{n} w_{i}x_{i}
$$

<br><br>
$$
x = 
\begin{bmatrix}
x_{1} \\\\
\vdots \\\\
x_{n}
\end{bmatrix}
, \quad
W = 
\begin{bmatrix}
w_{1,1} & \ldots & w_{1,n} \\\\
\vdots & \ddots & \vdots \\\\
w_{m,1} & \ldots & w_{m,n}
\end{bmatrix}
$$

$$Wx = 
\begin{bmatrix}
w_{1,1} & \ldots & w_{1,n} \\\\
\vdots & \ddots & \vdots \\\\
w_{m,1} & \ldots & w_{m,n}
\end{bmatrix}
\begin{bmatrix}
x_{1} \\\\
\vdots \\\\
x_{n}
\end{bmatrix} =
\begin{bmatrix}
\sum_{i=1}^{n} w_{1,i}x_{i} \\\\
\vdots \\\\
\sum_{i=1}^{n} w_{m,i}x_{i}
\end{bmatrix}
$$

### Vector and Matrix Gradients
The gradient of a scalar function with respect to a vector or matrix is:
The symbol $\frac{\sigma f}{\sigma x_ 1}$ means "partial derivative of f with respect to *x~1~*"

$$
\frac{\partial f}{\partial x} = 
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\\\
\vdots \\\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
,
\quad
\frac{\partial f}{\partial W} = 
\begin{bmatrix}
\frac{\partial f}{\partial w_{1,1}} & \cdots & \frac{\partial f}{\partial w_{1,n}} \\\\
\vdots & \ddots & \vdots \\\\
\frac{\partial f}{\partial w_{m,1}} & \cdots & \frac{\partial f}{\partial w_{m,n}}
\end{bmatrix}
$$


|![](https://www.researchgate.net/profile/Vladimir-Nasteski/publication/328146111/figure/fig4/AS:702757891751937@1544561946700/Visual-representation-of-the-linear-regression-22.ppm)|
|:-:|
|[© Vladimir Nasteski](https://www.researchgate.net/publication/328146111_An_overview_of_the_supervised_machine_learning_methods?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ)|

$$ f(x) = w^ T x + b = \sum_{j=0} ^{D-1} w_ j x_ j + b $$
- $f(x) = y$
- Generally, we want to choose the weights and bias, *w* and *b*, in order to minimize the errors.
- The errors are the vertical green bars in the figure at right, *&epsilon; = f(x) − y*.
- Some of them are positive, some are negative. What does it mean to "minimize" them?
    - $ f(x) = w^ T x + b = \sum_{j=0} ^{D-1} w_ j x_ j + b $
- Training token errors Using that notation, we can define a signed error term for every training token: *&epsilon; = f(x~i~) - y~i~*
- The error term is positive for some tokens, negative for other tokens. What does it mean to minimize it?

### Mean-squared error

Squared: tends to notice the big values and trying ignor small values.  

One useful criterion (not the only useful criterion, but perhaps the most common) of “minimizing the error” is to minimize the mean squared error:
$$  \mathcal{L} = \frac{1}{2n} \sum_{i=1}^ {n} \varepsilon_i^ 2 = \frac{1}{2n} \sum_{i=1}^ {n} (f(x_ i) - y_ i)^ 2  $$
The factor $\frac{1}{2}$ is included so that, so that when you differentiate ℒ , the 2 and the $\frac{1}{2}$ can cancel each other.

!!! note MSE = Parabola 
    Notice that MSE is a non -negative quadratic function of *f(**x**~i~) = **w**^T^ x~i~ + b*, therefore it’s a non negative quadratic function of ***w*** . Since it’s a non -negative quadratic function of ***w***, it has a unique minimum that you can compute in closed form! We won’t do that today. 
    $\mathcal{L} = \frac{1}{2n} \sum_{i=1}^ {n} (f(x_ i) - y_ i)^ 2$

### The iterative solution to linear regression (gradient descent):

- Instead of minimizing MSE in closed form, we’re going to use an iterative algorithm called gradient descent. It works like this:
    -  Start: random initial ***w*** and *b* (at *t=0*)
    - Adjust ***w*** and *b* to reduce MSE (*t=1*)
    - Repeat until you reach the optimum (*t = ∞*).

$ w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w} $
$ b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b} $


#### Finding the gradient

The loss function $ \mathcal{L} $ is defined as:
$$ \mathcal{L} = \frac{1}{2n} \sum_{i=1}^{n} L_i, \quad L_i = \varepsilon_i^2, \quad \varepsilon_i = w^T x_i + b - y_i $$
To find the gradient, we use the chain rule of calculus:
$$ \frac{\partial \mathcal{L}}{\partial w} = \frac{1}{2n} \sum_{i=1}^{n} \frac{\partial L_i}{\partial w}, \quad \frac{\partial L_i}{\partial w} = 2\varepsilon_i \frac{\partial \varepsilon_i}{\partial w}, \quad \frac{\partial \varepsilon_i}{\partial w} = x_i $$

Putting it all together,
$$ \frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} \varepsilon_i x_i $$


• Start from random initial values of $w$ and $b(at\ t= 0)$.
• Adjust $w$ and $b$ according to:

$$ w \leftarrow w - \frac{\eta}{n} \sum_{i=1}^{n} \varepsilon_i x_i $$
$$ b \leftarrow b - \frac{\eta}{n} \sum_{i=1}^{n} \varepsilon_i $$

#### Intuition:

- Notice the sign:
    - $ w \leftarrow w - \frac{\eta}{n} \sum_{i=1}^{n} \varepsilon_i x_i $
- If $ \varepsilon_i $ is positive ($ f(x_i) > y_i $), then we want to ==reduce== $ f(x_i) $, so we make $ w $ less like $ x_i $
- If $ \varepsilon_i $ is negative ($ f(x_i) < y_i $), then we want to ==increase== $ f(x_i) $, so we make $ w $ more like $ x_i $

### Gradient Descent

- If $n$ is large, computing or differentiating MSE can be expensive.
- The stochastic gradient descent algorithm picks one training token $(x_i, y_i)$ at random ("stochastically"), and adjusts $w$ in order to reduce the error a little bit for that one token:
  $$ w \leftarrow w - \eta \frac{\partial \mathcal{L}_i}{\partial w} $$
  ...where
  $$ \mathcal{L}_i = \varepsilon_i^2 = \frac{1}{2}(f(x_i) - y_i)^2 $$

### Stochastic gradient descent

$$
\mathcal{L}_i = \varepsilon_i^2 = \frac{1}{2}(w^T x_i + b - y_i)^2
$$

If we differentiate that, we discover that:

$$
\frac{\partial \mathcal{L}_i}{\partial w} = \varepsilon_i x_i,
\quad
\frac{\partial \mathcal{L}_i}{\partial b} = \varepsilon_i
$$

So the stochastic gradient descent algorithm is:

$$
w \leftarrow w - \eta \varepsilon_i x_i,
\quad
b \leftarrow b - \eta \varepsilon_i
$$

### Code Example



```python
import numpy as np
from functools import partial
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def updatew(x, y, W, b, e=0.01):
    E = np.sum(W*x +b - y)
    W -= np.sum(e*E*x)
    b -= e*E
    return W, b

slope = 1
intercept = 3
std_dev = 1
size = 100  # Size of the dataset

# Generate x values
X = np.random.uniform(low=-10, high=10, size=size)

# Generate y values based on the equation y = x + 3
# Add normal distributed noise with standard deviation of 0.4
Y = slope * X + intercept + np.random.normal(0, std_dev, size)

W = 0
b = 0
XX = []
for i in range(len(X)):
    W,b = updatew(X[i], Y[i], W, b, .01)
    XX +=[[W, b]]    

plt.plot(X, Y, 'o')
plt.plot(X, X*W + b)
plt.text(-9, 9, f'slop = {round(W, 2)}\nintercept = {round(b, 2)}')
plt.show()

# Your update function for the animation
def update(frame):
    # Update the data for the animated line plot, for example
    ln.set_data(X, X * XX[frame][0] + XX[frame][1] )
     # Update the text for the current frame
    txt.set_text(str(int(frame)) +': $y = {:.2f}x + {:.2f}$'.format(XX[frame][0], XX[frame][1]))
    return ln, txt
# Set up the figure and the line to animate
fig, ax = plt.subplots()
ln, = ax.plot([], [], 'r-', animated=True)
txt = ax.text(-9, 9, '', animated=True)  # Create a text object at (-9, 9)
# Plot the background points
ax.plot(X, Y, 'o')  # Static background points
# Init function to set up the background of each frame (if necessary)
def init():
    ax.set_xlim(min(X), max(X))
    ax.set_ylim(min(Y), max(Y))
    txt.set_text('')
    return ln,
# Create the animation
ani = FuncAnimation(fig, update, frames=100,
                    init_func=init, blit=True)
ani.save('animation_drawing.gif', writer='imagemagick', fps=10)
```

|![linear reguression](https://imgur.com/1O0dz1d.png)|
|:-:|

PS: In the old script, I wasn't fully understand how the linear regression works. So, I just made this script based on my personal understanding. So, you can find that I was using one dots from the set to calculate the loss and updates the $w$ and $b$ each time. But in real application situation, this way would make the updating process very noisy because single points could be very unreliable. According to the function above, we could use this function to update them:
- $ w \leftarrow w - \frac{\eta}{n} \sum_{i=1}^{n} \varepsilon_i x_i $
- $ b \leftarrow b - \frac{\eta}{n} \sum_{i=1}^{n} \varepsilon_i $

For using this function, two things you may like to change from the example: 

```python
# alter the weight and bias update function
def updatew(X, Y, W, b, e=0.01):
    E = W*X +b - Y
    W -= np.sum(E*X) * e/ len(X)
    b -= np.sum(E) * e/ len(X)
    return W, b

# starting the iteration and stop it manually
while True:
    W,b = updatew(X, Y, W, b, .05)
    print(W, b)
```

How do them different from each other?

||Single Points loss| Summed loss|
|:-|:-|:-|
|Last 5 rounds after so many iteration|<pre>1.1539 2.8111<br>1.1540 2.8105<br>1.2018 2.8054<br>1.1999 2.8021<br>0.9239 2.8350|<pre>0.9947 2.8171<br>0.9947 2.8171<br>0.9947 2.8171<br>0.9947 2.8171<br>0.9947 2.8171</pre>|
|Explained|Because the result from a single point has significant noise, the jitter can never be eliminated, making it difficult to achieve local optimization. | Because the loss is based on the entire dataset, it is very easy to achieve local optimization.|

This is why selecting an appropriate batch size during machine learning training is crucial. In the first example, the batch size is set to 1, while in the second, it equals the size of the training data. A very large batch size, such as using the entire dataset, can significantly increase the computational load, especially if the model is complex. Additionally, it may lead to the model getting stuck in local optima, which depends on the quality of the dataset. Therefore, choosing an appropriate batch size is essential for effective training.


---

## Perceptron

### Linear classifier: Definition

A linear classifier is defined by

$$
f(x) = \text{argmax } Wx + b
$$

where:

$$
Wx + b = 
\begin{bmatrix}
w_{1,1} & \ldots & w_{1,d} \\\\
\vdots & \ddots & \vdots \\\\
w_{v,1} & \ldots & w_{v,d}
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
\end{bmatrix}=
\begin{bmatrix}
w_{1}^T x + b_{1} \\\\
\vdots \\\\
w_{v}^T x + b_{v}
\end{bmatrix}
$$

$w_k, b_k$ are the weight vector and bias corresponding to class $k$, and the argmax function finds the element of the vector $wx$ with the largest value.

### Gradient descent


Suppose we have training tokens $(x_i, y_i)$, and we have some initial class vectors $w_1$ and $w_2$. We want to update them as

$$
w_1 \leftarrow w_1 - \eta \frac{\partial \mathcal{L}}{\partial w_1}
$$

$$
w_2 \leftarrow w_2 - \eta \frac{\partial \mathcal{L}}{\partial w_2}
$$

...where $\mathcal{L}$ is some loss function. What loss function makes sense?

## Transformation

Transformations can be a powerful tool for addressing various issues with model assumptions or improving the model fit. Applying transformations can help meet the necessary assumptions of linear regression, which are linearity, constant variance, normality of residuals, and independence of errors.

Common Transformations: **Log**, **Square Root**, **Inverse**, and **Box-Cox**


- **Box-Cox Transformation:**
  The Box-Cox transformation $ y_i^{(bc)} $ is defined as:
  $$
  y_i^{(bc)} = \begin{cases} 
    \frac{y_i^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\\\
    \ln(y_i) & \text{if } \lambda = 0 
  \end{cases}
  $$
  - This transformation is defined for $ y_i \geq 0 $.
  - It requires choosing an appropriate $ \lambda $, which can be done using statistical libraries that maximize the likelihood or minimize skewness.

- **Inverse Box-Cox Transformation:**
  The inverse transformation, which is used to revert the transformed data back to its original scale, is given by:
  $$
  y_i = \begin{cases} 
    \left( | \lambda y_i^{(bc)} + 1 |^{\frac{1}{\lambda}} \right) & \text{if } \lambda \neq 0 \\\\
    e^{y_i ^{(bc)}} & \text{if } \lambda = 0 
  \end{cases}
  $$
  - This transformation allows the original data values to be reconstructed from the transformed values, which is useful for interpretation and comparison with the original data scale after performing analyses or modeling on transformed data.


!!! note Choosing $ \lambda $**
    The value of $ \lambda $ can significantly affect the skewness and homoscedasticity of the residuals in regression modeling. It is usually selected to maximize the log-likelihood function of obtaining the transformed data under a normal distribution assumption or through cross-validation procedures.
    Tools like R and Python offer built-in functions to automatically find the optimal $ \lambda $ that best normalizes the data. For example, in R, the `boxcox` function from the MASS package can be used to estimate $ \lambda $.

### How to find the λ

1. Cross-validation
    - consider choices of at different scales, e.g.,
    λ ∈ {10^−4^, 10^−3^ ,10^−2^ ,10^−1^ ,1,10}
2. for each λ~i~,
    - iteratively build new random Fold from Training Set
        - fit Cross-Validation Train Set using
        - compute MSE for current Fold on Validation set
    - for each λ~i~ record average error σ and over all Folds
3. λ with smallest error - largest λ within one σ.

### Regularization 

**Regularization** involves adding a penalty to the loss function that a model minimizes. This penalty typically discourages complex models by imposing a cost on having larger parameter values, thereby promoting simpler, more robust models.

#### Types of Regularization


| **Type**         | **Description**                                           | **Effect**                        | **Equation**           |
| ------: | :----------------------------|:------- |:---: |
| L1 Regularization (Lasso) | Adds the absolute value of the magnitude of coefficients as a penalty term.   | Leads to sparse models where some weights are zero, effectively performing feature selection.  | $ L = \text{Loss} + \lambda \sum \|\beta_i\| $      |
| L2 Regularization (Ridge) | Adds the squared magnitude of coefficients as a penalty term.     | Distributes weight more evenly across features, less robust to outliers than L1.               | $ L = \text{Loss} + \lambda \sum \beta_i^2 $                  |
| Elastic Net      | Combination of L1 and L2 regularization.                                      | Balances feature selection and robustness, useful for correlated features. | $ L = \text{Loss} + \lambda_1 \sum \|\beta_i\|   + \lambda_2 \sum\beta_i^2 $                     |




#### Role of Regularization in Modeling

- **Preventing Overfitting**: By penalizing large coefficients, regularization reduces the model’s tendency to memorize training data, thus helping it to generalize better to new, unseen data.
- **Stability**: Regularization can make the learning algorithm more stable by smoothing the optimization landscape.
- **Handling Multicollinearity**: In regression, multicollinearity can make the model estimation unstable and sensitive to noise in the data. Regularization helps by constraining the coefficients path.

#### Regularization and $\lambda$

In regularization, $ \lambda $ (often called the regularization parameter) controls the strength of the penalty applied to the model. This is conceptually different from the $ \lambda $ used in transformations like Box-Cox, though both involve tuning a parameter to achieve better model behavior:

- In **Box-Cox transformations**, $ \lambda $ is selected to normalize the distribution of a variable or make relationships more linear.
- In **regularization**, $ \lambda $ adjusts the trade-off between fitting the training data well (low bias) and keeping the model parameters small (low variance).





## Performance

### Residuals and Standardized Residuals

Residuals: $e = y - \hat{y} $
Mean Square Error: $m = \frac{e^T e}{N}$


The equation you provided is for the standardized residual $ S_i $ in the context of regression analysis. Here's a breakdown of the equation and its components:


Standardized Residuals

$$
S_i = \frac{e_i}{\sigma} = \frac{e_i}{\sqrt{\frac{e^T e}{N} (1 - h_{i,i})}}
$$

- $ e_i $ is the residual for the $ i $-th observation, which is the difference between the observed value and the value predicted by the regression model.
- $ \sigma $ is the estimated standard deviation of the residuals.
- $ e^T e $ is the sum of squared residuals.
- $ N $ is the number of observations.
- $ h_{i,i} $ is the leverage of the $ i $-th observation, a measure from the hat matrix that indicates the influence of the $ i $-th observation on its own predicted value.

### Explanation:

1. **Standardized Residuals**: These are the residuals $ e_i $ normalized by an estimate of their standard deviation. Standardized residuals are useful for identifying outliers in regression analysis because they are scaled to have a variance of 1, except for their adjustment by the leverage factor.

2. **Leverage $ h_{i,i} $**: The leverage is a value that indicates how far an independent variable deviates from its mean. High leverage points can have an unduly large effect on the estimate of regression coefficients.

3. **Square Root Denominator**: The denominator of the standard deviation estimate involves the sum of squared residuals, which measures the overall error of the model, divided by the number of observations, adjusted for the leverage of each observation. This adjustment accounts for the fact that observations with higher leverage have a greater impact on the fit of the model and therefore should have less influence on the standardized residual.


### R2

$$
R^2 = \frac{\text{var}(\hat{y})}{\text{var}(y)}
$$

- $ \text{var}(y) = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2 $


### Cook’s distance

- **Coefficients and prediction with full training set**: $ y^{p} = X\hat{\beta} $
- **Coefficients and prediction excluding item i from training set**: $ y_i^{p} = X\hat{\beta}_i $
- **Cook's distance for point i:**
  $$ \text{Cook's distance} = \frac{(y^{p} - y_i^ {p})^ T (y^{p} - y_i^{p})}{dm} $$
  Where:
  - $d$ is the number of predictors (features) in the model, degree of freedom.
  - $m = \frac{e^T e}{N}$ is the mean squared error.
  - $N$ is the number of observations in the dataset.




<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
