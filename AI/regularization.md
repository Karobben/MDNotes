---
toc: true
url: regularization
covercopy: © Karobben
priority: 10000
date: 2024-12-30 10:50:04
title: Regularization
ytitle: Regularization
description:
excerpt: "Regularization is a way to make sure our model doesn't become too complicated. It ensures the model doesn’t overfit the training data while still making good predictions on new data. Think of it as adding a '<b>rule</b>' or '<b>constraint</b>' that prevents the model from relying too much on any specific feature or predictor."
tags: [Machine Learning, Data Science]
category: [Notes, Class, UIUC, AI]
cover: ""
thumbnail: ""
---

## Quick View

**Video Tutorial**:
- [StatQuest with Josh Starmer: Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30)
- [StatQuest with Josh Starmer: Regularization Part 2: Lasso (L1) Regression](https://www.youtube.com/watch?v=NGf0voTMlcs)
- [StatQuest with Josh Starmer: Regularization Part 3: Elastic Net Regression](https://www.youtube.com/watch?v=1dKRdX9bfIo)

### What is Regularization?

**Regularization** is a technique used in machine learning and regression to prevent **overfitting** by adding a penalty to the loss function. The penalty discourages overly complex models and large coefficients, helping the model generalize better to unseen data.

### Why Do We Need Regularization?

1. **Overfitting**: 
   - When a model becomes too complex, it memorizes the training data, leading to poor performance on test data.
   - Example: In polynomial regression, high-degree polynomials might perfectly fit the training data but fail to generalize.

2. **Ill-Conditioned Data**:
   - When predictors are highly correlated or there are many predictors relative to observations, the regression model can become unstable.

3. **Bias-Variance Tradeoff**:
   - Regularization introduces some bias but reduces variance, improving the model's robustness.

### Types of Regularization: Why Ridge, Lasso, and Elastic Net?

These are three popular regularization methods used for linear regression:


#### 1. **Ridge Regression (L2 Regularization)**:
- **Penalty**: Adds the squared magnitude of coefficients to the loss function.
$$
\text{Loss Function: } \sum_{i=1}^n (y_i - \hat{y}_ i)^2 + \lambda \sum_ {j=1}^p \beta_ j^2
$$
  - $ \lambda $: Regularization parameter (controls penalty strength).
  - $ \beta_j $: Coefficients of predictors.

- **Effect**:
  - Shrinks coefficients towards zero, but never makes them exactly zero.
  - Reduces the impact of less important predictors without removing them entirely.

- **Use Case**:
  - Works well when many predictors are correlated.


#### 2. **Lasso Regression (L1 Regularization)**:
- **Penalty**: Adds the absolute value of coefficients to the loss function.
$$
\text{Loss Function: } \sum_ {i=1}^n (y_ i - \hat{y}_ i)^2 + \lambda \sum_ {j=1}^p |\beta_ j|
$$

- **Effect**:
  - Can shrink some coefficients to exactly zero, effectively performing **feature selection**.
  - Helps in creating sparse models by keeping only the most relevant predictors.

- **Use Case**:
  - Useful when you expect only a subset of predictors to be important.


#### 3. **Elastic Net Regression**:
- **Penalty**: Combines both L1 (lasso) and L2 (ridge) penalties.
$$
\text{Loss Function: } \sum_{i=1}^n (y_ i - \hat{y}_ i)^2 + \lambda_ 1 \sum_{j=1}^p |\beta_ j| + \lambda_ 2 \sum_{j=1}^p \beta_ j^2
$$

- **Effect**:
  - Balances the strengths of Ridge and Lasso regression.
  - Retains the ability to perform feature selection (like Lasso) while handling multicollinearity (like Ridge).

- **Use Case**:
  - Best when there are many predictors and some are correlated, but feature selection is also desired.

### Comparison of Regularization Methods:

| **Method**       | **Penalty**          | **Effect on Coefficients**         | **Use Case**                                |
|-------------------|----------------------|------------------------------------|--------------------------------------------|
| **Ridge**         | $ \beta_j^2 $     | Shrinks coefficients, no zeros.    | Multicollinearity or many predictors.       |
| **Lasso**         | $\|\beta_j\|$     | Shrinks coefficients to zero.      | Feature selection with fewer predictors.    |
| **Elastic Net**   | $\|\beta_j\| + \beta_j^2 $ | Combination of Ridge and Lasso.    | Multicollinearity with feature selection.   |

### Why Are They Discussed Together?

- All three are **extensions of linear regression**.
- They **regularize the model** to prevent overfitting, but they differ in the type of penalty they impose on the coefficients.

## Ridge Regression

### Ridge Regression Loss Function
The Ridge regression modifies the Ordinary Least Squares (OLS) cost function by adding a penalty (regularization term) to the sum of squared coefficients:

$$
\text{Loss} = \sum_ {i=1}^n \left( y_ i - \hat{y}_ i \right)^2 + \lambda \sum_{j=1}^p \beta_ j^2
$$

Where:
- $ y_i $: Observed target value.
- $ \hat{y}_i $: Predicted value ($ \hat{y}_i = X_i \cdot \beta $).
- $ \beta_j $: Coefficients of the regression model.
- $ \lambda $: Regularization parameter (also called penalty parameter).

### Ridge Coefficient Solution

The Ridge regression coefficients are obtained by solving the following optimization problem:

$$
\min_{\beta} \\{ \\|y - X\beta\\|^2 + \lambda \\|\beta\\|^2  \\}
$$

1. **Matrix Form**:
   - Rewrite the problem in matrix notation:
     $$
     \min_{\beta} \\{ (y - X\beta)^T (y - X\beta) + \lambda \beta^T \beta \\}
     $$

2. **Solution for $ \beta $**:
   - Differentiating the loss function with respect to $ \beta $, we get:
     $$
     \beta = \left( X^T X + \lambda I \right)^{-1} X^T y
     $$
   - Here:
     - $ X^T X $: Correlation matrix of predictors.
     - $ \lambda I $: Regularization term, where $ I $ is the identity matrix.
     - $ \lambda $: Controls the trade-off between minimizing the squared error and penalizing large coefficients.

### Why Add $ \lambda I $?
- Inverse of $ X^T X $ might not exist if the predictors are highly correlated or there are fewer observations than predictors (multicollinearity).
- Adding $ \lambda I $ ensures that $ X^T X + \lambda I $ is always invertible.

### Finding the Optimal $ \lambda $

1. **Grid Search with Cross-Validation**:
   - Evaluate the model's performance (e.g., Mean Squared Error) for different values of $ \lambda $.
   - Use k-fold cross-validation to select the $ \lambda $ that minimizes validation error.

2. **Mathematical Insight**:
   - When $ \lambda = 0 $: Ridge reduces to Ordinary Least Squares (OLS).
   - As $ \lambda \to \infty $: Coefficients $ \beta \to 0 $ (model becomes very simple).

3. **Validation-Based Optimization**:
   - Define a range of $ \lambda $ values (e.g., $ \lambda = [0.001, 0.01, 0.1, 1, 10, 100] $).
   - For each $ \lambda $, perform cross-validation and select the value with the lowest error.

### Example: Finding $ \lambda $ with Cross-Validation

Here’s Python code to find the optimal $ \lambda $ using grid search:

```python
from sklearn.model_selection import cross_val_score

# Define a range of lambda (alpha) values
alphas = np.logspace(-3, 3, 50)  # Lambda values from 0.001 to 1000
# Compute cross-validated MSE and standard deviation for each alpha
mse_values = []
std_errors = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = -cross_val_score(ridge, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
    mse_values.append(scores.mean())
    std_errors.append(scores.std())

# Plot the results with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(alphas, mse_values, yerr=std_errors, fmt='o', linestyle='-', label='Cross-validated MSE', capsize=3)
plt.xscale('log')  # Log scale for alpha
plt.xlabel("Lambda (α)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Finding Optimal Lambda (α) for Ridge Regression with Error Bars")
plt.axvline(alphas[np.argmin(mse_values)], color='red', linestyle='--', label=f"Optimal λ = {alphas[np.argmin(mse_values)]:.3f}")
plt.legend()
plt.grid(True)
plt.show()

# Output optimal lambda
optimal_lambda_error_bar = alphas[np.argmin(mse_values)]
optimal_lambda_error_bar
```

<pre>
0.21209508879201902
</pre>

![](https://imgur.com/ir98xgb.png)

### Key Insights
1. **Ridge Regression Purpose**:
   - Penalizes large coefficients to reduce model complexity and improve generalization.

2. **Finding $ \lambda $**:
   - Perform grid search with cross-validation to select $ \lambda $ that minimizes validation error.



```python
# Simulate noisier data
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 2)  # Two predictors
X[:, 1] = X[:, 0] + np.random.normal(0, 0.1, size=n_samples)  # Add stronger multicollinearity
y = 4 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 2, size=n_samples)  # More noise in the data

# Randomly sample 20% for training
train_indices = np.random.choice(range(n_samples), size=int(0.2 * n_samples), replace=False)
test_indices = [i for i in range(n_samples) if i not in train_indices]

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# Ordinary Least Squares Regression
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)
y_pred_ols = ols_model.predict(X_test)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)  # Alpha is equivalent to λ
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate models
mse_ols = mean_squared_error(y_test, y_pred_ols)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Return updated MSE and coefficients
mse_results_updated = {
    "Mean Squared Error (OLS)": mse_ols,
    "Mean Squared Error (Ridge)": mse_ridge,
    "OLS Coefficients": ols_model.coef_,
    "Ridge Coefficients": ridge_model.coef_,
}

mse_results_updated


# Correct the regression line plotting using predicted results

# Generate predictions for the entire feature range for consistent straight lines
X_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
X_range_full = np.hstack([X_range, X_range + np.random.normal(0, 0.1, size=X_range.shape)])

# Predict the regression lines for OLS and Ridge models
y_ols_line = ols_model.predict(X_range_full)
y_ridge_line = ridge_model.predict(X_range_full)

# Plot regression results
plt.figure(figsize=(14, 6))

# OLS Regression
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], y_test, color='blue', label="Test Set", alpha=0.7)
plt.scatter(X_train[:, 0], y_train, color='orange', label="Training Set", alpha=0.7)
plt.plot(X_range[:, 0], y_ols_line, color='red', label="OLS Regression Line")
plt.title("OLS Regression")
plt.xlabel("Feature 1")
plt.ylabel("Target (y)")
plt.legend()

# Ridge Regression
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], y_test, color='blue', label="Test Set", alpha=0.7)
plt.scatter(X_train[:, 0], y_train, color='orange', label="Training Set", alpha=0.7)
plt.plot(X_range[:, 0], y_ridge_line, color='purple', label="Ridge Regression Line")
plt.title("Ridge Regression")
plt.xlabel("Feature 1")
plt.ylabel("Target (y)")
plt.legend()

plt.tight_layout()
plt.show()

# Plot feature contributions (coefficients) as bar plots
plt.figure(figsize=(10, 6))
x_labels = ['Feature 1', 'Feature 2']

# OLS Coefficients
plt.bar(x_labels, ols_model.coef_, label='OLS Coefficients', alpha=0.7, color='red')

# Ridge Coefficients
plt.bar(x_labels, ridge_model.coef_, label='Ridge Coefficients', alpha=0.7, color='blue')

# Add title and labels
plt.title("Comparison of Feature Contributions (OLS vs Ridge)")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()
```

<pre>
{'Mean Squared Error (OLS)': 3.747535481239866,
 'Mean Squared Error (Ridge)': 3.7344119726941427,
 'OLS Coefficients': array([4.0641917 , 3.31246222]),
 'Ridge Coefficients': array([2.93270253, 2.91932805])}
</pre>

![](https://imgur.com/tUDhjah.png)
![](https://imgur.com/dJzQZyh.png)

In this specific example, ridge regression slight reduced the mean squared error by reducing the contribution of **feature 1**. Contribution of the **feature 1** and **feature 2** are almost the same (Blue color in barplot). The linear regression plot was updated by removing the effects of **feature 2**.


## Compare 3 Methods

Code continue from above
```python
from sklearn.linear_model import ElasticNet

# Perform Elastic Net Regression
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Alpha controls regularization strength, l1_ratio balances Lasso and Ridge
elastic_net_model.fit(X_train, y_train)
y_pred_elastic_net = elastic_net_model.predict(X_test)

# Evaluate Elastic Net model
mse_elastic_net = mean_squared_error(y_test, y_pred_elastic_net)

# Plot feature contributions (coefficients) for Elastic Net
plt.figure(figsize=(10, 6))
plt.bar(x_labels, ols_model.coef_, label="OLS Coefficients", alpha=0.7, color="red")
plt.bar(x_labels, ridge_model.coef_, label="Ridge Coefficients", alpha=0.7, color="blue")
plt.bar(x_labels, lasso_model.coef_, label="Lasso Coefficients", alpha=0.7, color="green")
plt.bar(x_labels, elastic_net_model.coef_, label="Elastic Net Coefficients", alpha=0.7, color="purple")
plt.title("Comparison of Feature Contributions (OLS, Ridge, Lasso, Elastic Net)")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()

# Return MSE and coefficients for Elastic Net
mse_results_elastic_net = {
    "Mean Squared Error (OLS)": mse_ols,
    "Mean Squared Error (Ridge)": mse_ridge,
    "Mean Squared Error (Lasso)": mse_lasso,
    "Mean Squared Error (Elastic Net)": mse_elastic_net,
    "OLS Coefficients": ols_model.coef_,
    "Ridge Coefficients": ridge_model.coef_,
    "Lasso Coefficients": lasso_model.coef_,
    "Elastic Net Coefficients": elastic_net_model.coef_,
}

mse_results_elastic_net
```

<pre>
{'Mean Squared Error (OLS)': 3.747535481239866,
 'Mean Squared Error (Ridge)': 3.7344119726941427,
 'Mean Squared Error (Lasso)': 3.681549054209485,
 'Mean Squared Error (Elastic Net)': 3.810867636824328,
 'OLS Coefficients': array([4.0641917 , 3.31246222]),
 'Ridge Coefficients': array([2.93270253, 2.91932805]),
 'Lasso Coefficients': array([3.3579283 , 2.97769008]),
 'Elastic Net Coefficients': array([2.72149455, 2.71825096])}
</pre>

![](https://imgur.com/ExhrW4P.png)

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
