---
toc: true
url: linearoptimal
covercopy: © Karobben
priority: 10000
date: 2024-12-30 14:59:30
title: Linear Model Optimization
ytitle: Linear Model Optimization
description: "Linear Model Optimization"
excerpt: "Linear Model Optimization"
tags: [Machine Learning, Data Science]
category: [Notes, Class, UIUC, AI]
cover: ""
thumbnail: ""
---

## Measure Information 

Both Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) are measures used to evaluate the quality of statistical models, particularly in the context of selecting the best model size or complexity.

1. **Why Use AIC and BIC?**
   - When building statistical or machine learning models, we often face the challenge of balancing **model fit** (how well the model explains the data) with **model simplicity** (avoiding overfitting).
   - AIC and BIC are metrics that help in selecting the best model by incorporating penalties for the number of parameters used in the model.

2. **Akaike Information Criterion (AIC):**
   - AIC estimates the relative quality of a model for a given dataset.
   - Formula:
     $$
     \text{AIC} = 2k - 2 \ln(L)
     $$
     - $ k $: Number of parameters in the model.
     - $ L $: Likelihood of the model (how well it fits the data).
   - **Objective**: Choose the model with the **lowest AIC value**, which balances fit and complexity.

3. **Bayesian Information Criterion (BIC):**
   - Similar to AIC, BIC adds a stronger penalty for model complexity to account for overfitting.
   - Formula:
     $$
     \text{BIC} = k \ln(n) - 2 \ln(L)
     $$
     - $ n $: Number of observations in the dataset.
     - $ k $: Number of parameters in the model.
   - **Objective**: Choose the model with the **lowest BIC value** for a balance between fit and simplicity, especially when sample size $ n $ is large.

4. **Key Difference Between AIC and BIC:**
   - AIC focuses on model quality and is less strict about model size.
   - BIC penalizes complexity more heavily, making it more conservative in selecting simpler models.


**Applications**:
- Model selection in regression, time-series analysis, and machine learning.
- Comparing models with different numbers of features or parameters.
- Evaluating trade-offs between underfitting and overfitting.

Would you like a detailed example or visual demonstration of how AIC and BIC are used?



| **Criterion**          | **Formula**                          | **Focus**                         | **Penalty for Complexity**        | **Use Case**                                      | **Objective**                     |
|-------------------------|---------------------------------------|------------------------------------|------------------------------------|--------------------------------------------------|------------------------------------|
| **Akaike Information Criterion (AIC)** | $ 2k - 2\ln(L) $               | Model fit vs. simplicity           | Proportional to $ k $            | Choosing models that balance goodness-of-fit and simplicity | Minimize AIC                      |
| **Bayesian Information Criterion (BIC)** | $ k\ln(n) - 2\ln(L) $          | Model fit vs. parsimony            | Stronger penalty with $ \ln(n) $ | Suitable for large datasets and emphasizing simpler models   | Minimize BIC                      |
| **Penalty Strength**    | Moderate                            | High                              | **Depends on Sample Size ($ n $)** | Larger datasets lead to stricter penalties in BIC            |                                 |
| **Common Application**  | Time-series, regression, machine learning | Model selection across varying complexity | Multi-model comparison             | Best when balancing underfitting and overfitting             |

1. **AIC**:
   - Prefers models with a better balance between complexity and fit.
   - Less conservative than BIC, suitable for small datasets or exploratory analysis.

2. **BIC**:
   - Stronger emphasis on simplicity.
   - More appropriate for larger datasets or when avoiding overfitting is crucial.

3. **Choosing Between AIC and BIC**:
   - Use **AIC** if you prioritize model quality over strict simplicity.
   - Use **BIC** if simplicity and generalization are more important.

### Likelihood

When calculating AIC or BIC, the likelihood refers to **how well the model trained on the training data** explains the same training data. The likelihood is not calculated on the test data, as AIC and BIC are measures of model quality on the training dataset itself.

### Likelihood in AIC/BIC Context:

1. **Training Data**:
   - We use the model parameters (e.g., coefficients in regression) estimated from the training data to calculate the likelihood of the training data.
2. **Likelihood Calculation**:
   - For a model trained on the training data, the likelihood is the probability (or density) of the observed training data under the model:
     $$
     L(\theta | \text{Training Data}) = \prod_{i=1}^n f(y_i | \theta)
     $$
     Where:
     - $ y_i $: Observed target value.
     - $ \theta $: Model parameters estimated during training.
     - $ f(y_i | \theta) $: Probability density of $ y_i $ under the model.
3. **Log-Likelihood for AIC/BIC**:
   - Instead of working with $ L $, we calculate the **log-likelihood** to simplify computations:
     $$
     \ln L(\theta | \text{Training Data}) = \sum_{i=1}^n \ln f(y_i | \theta)
     $$

### Steps to Calculate Likelihood for AIC/BIC:

1. Train the Model:
   - Use the training data to estimate the model parameters ($ \theta $).
2. Calculate Predictions ($ \hat{y}_i $):
   - Predict the mean or central tendency of the model for each training data point.
3. Calculate Residuals and Likelihood:
   - Assume a distribution for the residuals (e.g., normal distribution).
   - For a normal distribution:
     $$
     f(y_i | \hat{y}_i, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{y}_i)^ 2}{2\sigma^ 2}\right)
     $$
   - The log-likelihood becomes:
     $$
     \ln L = \sum_{i=1}^n \left[ -\frac{1}{2} \ln(2\pi\sigma^2) - \frac{(y_i - \hat{y}_i)^ 2}{2\sigma^ 2} \right]
     $$

$\sigma$ represents the **standard deviation of the residuals**

### Example: Using Training Data to Calculate Likelihood


```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Simulated training data
X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_train = np.array([1.2, 2.3, 2.8, 4.1, 5.3])

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)

# Calculate residuals and variance
residuals = y_train - y_pred_train
sigma_squared = np.var(residuals, ddof=1)  # Variance of residuals

# Calculate log-likelihood
n = len(y_train)
log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - np.sum((residuals**2) / (2 * sigma_squared))

# AIC and BIC
k = 2  # Number of parameters (intercept + slope)
aic = 2 * k - 2 * log_likelihood
bic = k * np.log(n) - 2 * log_likelihood

{"Log-Likelihood": log_likelihood, "AIC": aic, "BIC": bic}
```

### Example

Source: [Model selection with AIC and AICc](https://www.youtube.com/watch?v=HOqHI53x9Go)
![](https://imgur.com/K4j8QGS.png)

## Forward stagewise regression and Backward stagewise regression

**Backward stagewise regression** and **Forward stagewise regression** are methods for variable selection and model fitting, primarily used in regression contexts. They are stepwise procedures for adding or removing predictors in a systematic way to improve model performance or interpretability.

### **Backward Stagewise Regression**

#### Overview:
- Starts with a **full model** (all predictors included).
- Gradually **removes predictors** one by one, based on a criterion (e.g., p-value, AIC, or adjusted $ R^2 $).
- The goal is to find a smaller, simpler model without significantly compromising the fit.

#### Procedure:
1. Begin with a model containing all predictors.
2. Evaluate the significance of each predictor (e.g., using p-values).
3. Remove the **least significant predictor** (highest p-value) that exceeds a predefined

threshold (e.g., $p > 0.05$).

4. Refit the model and repeat the process until all remaining predictors are statistically significant or meet the stopping criteria.

#### Advantages:
- Simple and interpretable.
- Useful for removing irrelevant predictors in high-dimensional datasets.

#### Disadvantages:
- Can miss optimal combinations of predictors.
- Sensitive to multicollinearity among predictors.

### 2. **Forward Stagewise Regression**

#### Overview:
- Starts with an **empty model** (no predictors included).
- Gradually **adds predictors** one at a time, based on a criterion (e.g., reducing residual sum of squares or improving AIC/BIC).
- The goal is to build a model step-by-step, adding only significant predictors.

#### Procedure:
1. Begin with an empty model.
2. Evaluate all predictors not yet in the model, adding the one that most improves the model fit (e.g., the one with the smallest p-value or largest improvement in $ R^2 $).
3. Refit the model and repeat the process until no additional predictors meet the inclusion criteria.

#### Advantages:
- Can handle datasets with a large number of predictors.
- Less likely to overfit compared to starting with a full model.

#### Disadvantages:
- Ignores potential joint effects of predictors (e.g., interactions).
- May miss the best subset of predictors.

### Key Differences Between Backward and Forward Stagewise Regression

| Feature                     | Backward Stagewise                 | Forward Stagewise                  |
|-----------------------------|-------------------------------------|-------------------------------------|
| **Starting Point**          | Full model (all predictors).       | Empty model (no predictors).       |
| **Procedure**               | Removes predictors iteratively.    | Adds predictors iteratively.       |
| **Use Case**                | Small datasets with fewer predictors. | Large datasets with many predictors. |
| **Limitations**             | May retain redundant predictors.   | May miss joint effects of predictors. |


### When to Use Each Method?

- **Backward Stagewise**:
  - When you suspect many predictors are irrelevant.
  - When computational resources are not a concern (since fitting starts with a large model).

- **Forward Stagewise**:
  - When you have a large number of predictors and computational efficiency is critical.
  - When you want a simpler starting point and add complexity gradually.

### A Quick Example

```python
# Re-import necessary libraries after environment reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate a dataset with 100 samples and 10 features
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=10, noise=10, random_state=42)

# Initialize model and variables for Forward Stagewise Regression
selected_features = []
remaining_features = list(range(X.shape[1]))
forward_scores = []

# Forward Stagewise Regression
for _ in range(len(remaining_features)):
    scores = []
    for feature in remaining_features:
        # Fit a model with the current feature added
        features_to_test = selected_features + [feature]
        model = LinearRegression().fit(X[:, features_to_test], y)
        score = model.score(X[:, features_to_test], y)  # R^2 score
        scores.append((score, feature))
    
    # Select the feature with the highest R^2 score
    scores.sort(reverse=True)
    best_score, best_feature = scores[0]
    forward_scores.append(best_score)
    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

# Results of Forward Stagewise Regression
selected_features_forward = selected_features  # Save selected features for clarity

# Backward Stagewise Regression
selected_features_backward = list(range(X.shape[1]))
backward_scores = []

for _ in range(len(selected_features_backward) - 1):
    scores = []
    for feature in selected_features_backward:
        # Fit a model with the current feature removed
        features_to_test = [f for f in selected_features_backward if f != feature]
        model = LinearRegression().fit(X[:, features_to_test], y)
        score = model.score(X[:, features_to_test], y)  # R^2 score
        scores.append((score, feature))
    
    # Remove the feature with the smallest impact on R^2 score
    scores.sort(reverse=True)
    best_score, worst_feature = scores[-1]
    backward_scores.append(best_score)
    selected_features_backward.remove(worst_feature)

# Plot R^2 scores for Forward and Backward Stagewise Regression
plt.figure(figsize=(12, 6))

# Forward Stagewise Regression
plt.plot(range(1, len(forward_scores) + 1), forward_scores, marker='o', label='Forward Stagewise', color='blue')

# Backward Stagewise Regression
plt.plot(range(len(backward_scores), 0, -1), backward_scores, marker='o', label='Backward Stagewise', color='red')

# Formatting the plot
plt.title("R² Scores During Forward and Backward Stagewise Regression")
plt.xlabel("Number of Features")
plt.ylabel("R² Score")
plt.xticks(range(1, len(forward_scores) + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<script type="text/javascript">
  google.charts.load('current', { packages: ['corechart'] });
  google.charts.setOnLoadCallback(drawChart);

  function drawChart() {
    var data = google.visualization.arrayToDataTable([
          ['Number of Features', 'Forward Stagewise R²', 'Backward Stagewise R²'],
          [1, 0.287, null],
          [2, 0.511, null],
          [3, 0.698, null],
          [4, 0.801, null],
          [5, 0.918, null],
          [6, 0.988, null],
          [7, 0.995, null],
          [8, 0.997, null],
          [9, 0.997, null],
          [9, 0.997, 0.787],
          [8, null, 0.643],
          [7, null, 0.441],
          [6, null, 0.285],
          [5, null, 0.176],
          [4, null, 0.076],
          [3, null, 0.045],
          [2, null, 0.011],
          [1, null, 0.002],
        ]);

    var options = {
      title: 'R² Scores During Forward and Backward Stagewise Regression',
      hAxis: { title: 'Number of Features' },
      vAxis: { title: 'R² Score', minValue: 0, maxValue: 1 },
      legend: { position: 'top' },
      colors: ['blue', 'red'],
    };

    var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
    chart.draw(data, options);
  }
</script>
 
<div id="chart_div" style="width: 100%  ; height: 300px"></div>

### Limitations

**Forward and Backward Stagewise Regression** can become computationally expensive and impractical when dealing with a **large number of features (e.g., 1000+ features)** because:

1. **High Computational Cost**:
   - Both methods involve iteratively adding or removing features, which requires fitting a model at each step. For large datasets, this becomes infeasible.
2. **Potential Overfitting**:
   - With a large number of features, stepwise methods might select features that fit noise in the data rather than actual patterns.
3. **Ignoring Interactions**:
   - These methods do not account for interactions between features, which can lead to suboptimal feature selection.


**Alternative Methods for Large Feature Spaces**

| **Method**                 | **Description**                                                                                   | **Advantages**                                                                                     | **Disadvantages**                                                                                   | **Best Use Case**                                                                                  |
|----------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Lasso Regression (L1)**  | Shrinks coefficients and sets some to exactly zero for feature selection.                        | - Efficient for high-dimensional data.<br>- Automatically selects features.<br>- Prevents overfitting. | - May ignore correlated features.<br>- Requires hyperparameter tuning ($ \lambda $).             | When many features are irrelevant, and sparse solutions are desired.                              |
| **Elastic Net**            | Combines L1 (Lasso) and L2 (Ridge) regularization.                                               | - Balances feature selection and handling multicollinearity.<br>- Suitable for correlated features. | - More complex than Lasso.<br>- Requires tuning of both $ \lambda $ and $ \alpha $.            | When predictors are highly correlated, and feature selection is needed.                           |
| **Recursive Feature Elimination (RFE)** | Iteratively removes the least important features based on a chosen model.                                    | - Works with any estimator (e.g., linear, tree-based).<br>- Provides a rank of feature importance. | - Computationally expensive.<br>- Sensitive to model choice and training data.                     | When model-specific feature ranking is required.                                                   |
| **Principal Component Analysis (PCA)** | Reduces dimensionality by transforming features into uncorrelated components that capture most variance.     | - Handles high-dimensional data well.<br>- Removes multicollinearity.<br>- No need for target variable. | - Components are linear combinations of features, losing interpretability.<br>- Not ideal for feature selection. | When reducing dimensionality is more important than interpretability.                              |
| **Tree-Based Feature Importance** | Uses models like Random Forest or Gradient Boosting to rank feature importance.                                | - Naturally handles non-linearity.<br>- Accounts for feature interactions.<br>- Fast for large datasets. | - Can be biased toward high-cardinality features.<br>- Does not directly reduce feature count.      | When using tree-based models or ranking feature importance is a priority.                          |
| **Mutual Information**     | Measures the statistical dependency between features and the target variable.                   | - Non-parametric.<br>- Detects non-linear relationships.                                            | - Computationally expensive for many features.<br>- Does not handle feature interactions.           | When quantifying feature relevance to the target variable without assumptions is needed.           |
| **Feature Clustering**     | Groups similar features into clusters and uses cluster representatives for modeling.             | - Reduces redundancy in correlated features.<br>- Scales well with high-dimensional data.           | - May lose specific feature contributions.<br>- Requires a meaningful distance metric.             | When dealing with highly correlated features or datasets with groups of similar features.           |
| **Embedding-Based Methods** | Uses deep learning or models like word2vec to transform features into a lower-dimensional space. | - Captures complex relationships between features.<br>- Flexible for large feature spaces.         | - Requires advanced techniques and computational resources.<br>- May lose interpretability.         | When handling very high-dimensional data (e.g., text, genomic data) with complex dependencies.      |


#### Recommendations:

- **Lasso Regression**: If feature selection is the goal and the data has many irrelevant features.
- **Elastic Net**: If features are highly correlated and Lasso alone may struggle.
- **PCA**: When interpretability is less important, and you want to reduce dimensionality.
- **Tree-Based Importance**: For datasets where feature importance ranking is needed, especially with tree-based models.
- **Feature Clustering**: For correlated features where redundancy needs to be reduced.

## M-Estimators

**M-Estimators** (Maximum Likelihood-type Estimators) are a general class of estimators in statistics used for robust parameter estimation. They extend the principle of Maximum Likelihood Estimation (MLE) to allow for more flexibility and robustness, especially in the presence of outliers or non-normal errors.

### What Are M-Estimators?

1. **Definition**:
   - M-Estimators generalize Maximum Likelihood Estimators by minimizing a **loss function** (also called the objective function) over the parameters of interest.

2. **Loss Function**:
   - The core idea is to minimize a function of residuals:
     $$
     \hat{\theta} = \arg\min_{\theta} \sum_{i=1}^n \rho\left(\frac{r_i}{\sigma}\right)
     $$
     Where:
     - $ r_i = y_i - f(x_i, \theta) $: Residual (difference between observed and predicted values).
     - $ \rho(\cdot) $: A loss function that determines the contribution of residuals.
     - $ \sigma $: Scale parameter (controls the spread).

3. **Goal**:
   - Instead of focusing purely on minimizing squared residuals (like in Ordinary Least Squares), M-Estimators allow for more flexible functions to make the estimator **less sensitive to outliers**.

### Examples of M-Estimators

| **Type**                     | **Loss Function ($ \rho $)**                   | **Characteristics**                                          |
|-------------------------------|------------------------------------------------|-------------------------------------------------------------|
| **Ordinary Least Squares (OLS)** | $ \rho( r ) = r^2 $                           | Highly sensitive to outliers. Minimizes sum of squared errors. |
| **Huber Loss**                | $\rho( r) = \begin{cases} r^2 & \text{if } \|r\| \leq c \\\\ 2c\|r\| - c^2 & \text{if } \|r\| > c\end{cases}$   | Combines squared loss (for small residuals) and absolute loss (for large residuals). |
| **Tukey's Biweight**          | $\rho( r ) = \begin{cases}  c^2\left(1 - \left[1 - \left(\frac{r}{c}\right)^ 2\right]^ 3\right) & \text{if } \|r\| \leq c \\\\ c^2 & \text{if } \|r\| > c \end{cases}$                                     | Completely ignores residuals larger than a threshold $ c $. |
| **Huberized Absolute Loss**   | $\rho( r) = \|r\|$                            | Linear penalty, robust but less efficient.                  |

### Advantages of M-Estimators

1. **Robustness to Outliers**
2. **Flexibility**
3. **Generalization of MLE**:
   - MLE is a special case of M-Estimators, making them widely applicable in parametric settings.

### When to Use M-Estimators?

1. **Presence of Outliers**:
2. **Non-Normal Errors**:
3. **Heavy-Tailed Distributions**:


### Practical Example: Using Huber Loss

Below is an example of applying **Huber Loss** to regression in Python:

```python
from sklearn.linear_model import HuberRegressor
import numpy as np
import matplotlib.pyplot as plt

# Simulate data with outliers
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.flatten() + np.random.normal(0, 1, size=X.shape[0])
y[::10] += 20  # Add outliers every 10th point

# Fit Ordinary Least Squares (OLS) Regression
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X, y)

# Fit Huber Regression
huber = HuberRegressor(epsilon=1.35).fit(X, y)

# Plot the results
plt.scatter(X, y, color="blue", label="Data with Outliers")
plt.plot(X, ols.predict(X), color="red", label="OLS Regression Line")
plt.plot(X, huber.predict(X), color="green", label="Huber Regression Line")
plt.title("Comparison of OLS and Huber Regression")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```


<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<script type="text/javascript">
  google.charts.load('current', { packages: ['corechart'] });
  google.charts.setOnLoadCallback(drawChart);

  function drawChart() {
    // Data for OLS and Huber Regression
    var data = google.visualization.arrayToDataTable([['X', 'Observed Data', 'OLS Regression', 'Huber Regression'],
 [0.0, 20.496714153011233, 2.3618406112691552, -0.05895399071098967], [0.10101010101010101, 0.16476600185911838, 2.6554630935183825, 0.24599617373579663], [0.20202020202020202, 1.2537491441612985, 2.9490855757676098, 0.550946338182583], [0.30303030303030304, 2.4321207654989347, 3.242708058016837, 0.8558965026293694], [0.40404040404040403, 0.9779678373978762, 3.5363305402660643, 1.1608466670761555], [0.5050505050505051, 1.2810145582023347, 3.8299530225152916, 1.465796831522942], [0.6060606060606061, 3.39739463368921, 4.123575504764519, 1.7707469959697284], [0.7070707070707071, 2.88864685036503, 4.417197987013745, 2.0756971604165146], [0.8080808080808081, 1.9547680383074721, 4.710820469262973, 2.380647324863301], [0.9090909090909091, 3.2698327708586916, 5.0044429515122, 2.6855974893100876], [1.0101010101010102, 22.566885337490568, 5.298065433761428, 2.990547653756874], [1.1111111111111112, 2.8676035797630766, 5.591687916010654, 3.29549781820366], [1.2121212121212122, 3.878325907929671, 5.8853103982598824, 3.6004479826504467], [1.3131313131313131, 2.0261136947361416, 6.178932880509109, 3.905398147097233], [1.4141414141414141, 2.5175064099112094, 6.472555362758336, 4.210348311544019], [1.5151515151515151, 3.983167016213572, 6.766177845007563, 4.5152984759908055], [1.6161616161616161, 3.835653728150425, 7.059800327256791, 4.820248640437591], [1.7171717171717171, 5.465762484110425, 7.353422809506018, 5.125198804884378], [1.8181818181818181, 4.546521379024243, 7.647045291755245, 5.430148969331165], [1.9191919191919191, 4.345272056240466, 7.940667774004472, 5.73509913377795], [2.0202020202020203, 27.526254829527616, 8.2342902562537, 6.040049298224737], [2.121212121212121, 6.1378600631498275, 8.527912738502927, 6.344999462671523], [2.2222222222222223, 6.734194871354591, 8.821535220752153, 6.64994962711831], [2.323232323232323, 5.5449487834835125, 9.11515770300138, 6.954899791565095], [2.4242424242424243, 6.728344548202091, 9.40878018525061, 7.259849956011883], [2.525252525252525, 7.6866801654674415, 9.702402667499836, 7.564800120458668], [2.6262626262626263, 6.727794301365576, 9.996025149749062, 7.8697502849054555], [2.727272727272727, 8.557516200163853, 10.289647631998289, 8.17470044935224], [2.8282828282828283, 7.88420979492968, 10.583270114247517, 8.479650613799027], [2.929292929292929, 8.49618503808551, 10.876892596496743, 8.784600778245814], [3.0303030303030303, 28.489202478679694, 11.170515078745971, 9.0895509426926], [3.131313131313131, 11.24621757844833, 11.464137560995198, 9.394501107139385], [3.2323232323232323, 9.683472472231763, 11.757760043244426, 9.699451271586172], [3.3333333333333335, 8.9422890710441, 12.051382525493654, 10.00440143603296], [3.4343434343434343, 11.125575215133491, 12.34500500774288, 10.309351600479745], [3.5353535353535355, 9.385216956089582, 12.638627489992109, 10.614301764926532], [3.6363636363636362, 11.117954504095664, 12.932249972241335, 10.919251929373319], [3.7373737373737375, 9.252451088241438, 13.225872454490563, 11.224202093820105], [3.8383838383838382, 10.186965466253085, 13.51949493673979, 11.52915225826689], [3.9393939393939394, 12.015043054050942, 13.813117418989018, 11.834102422713677], [4.040404040404041, 32.85967870120753, 14.106739901238244, 12.139052587160464], [4.141414141414141, 12.595610705432392, 14.40036238348747, 12.444002751607249], [4.242424242424242, 12.611624444884486, 14.693984865736699, 12.748952916054035], [4.343434343434343, 12.729199334713742, 14.987607347985925, 13.053903080500822], [4.444444444444445, 11.854811342965906, 15.281229830235153, 13.358853244947609], [4.545454545454545, 12.916519427968927, 15.57485231248438, 13.663803409394394], [4.646464646464646, 13.47875516843415, 15.868474794733606, 13.96875357384118], [4.747474747474747, 15.299546468643157, 16.162097276982834, 14.273703738287967], [4.848484848484849, 14.889072835023008, 16.45571975923206, 14.578653902734755], [4.94949494949495, 13.085444693122115, 16.74934224148129, 14.883604067181542], [5.05050505050505, 35.47559912090995, 17.042964723730513, 15.188554231628325], [5.151515151515151, 15.069463174129137, 17.336587205979743, 15.493504396075112], [5.252525252525253, 15.080653757269799, 17.630209688228973, 15.7984545605219], [5.353535353535354, 16.67228234944693, 17.9238321704782, 16.103404724968687], [5.454545454545454, 17.394635886132313, 18.217454652727426, 16.40835488941547], [5.555555555555555, 17.597946785782863, 18.511077134976652, 16.713305053862257], [5.656565656565657, 16.13047944647433, 18.80469961722588, 17.018255218309044], [5.757575757575758, 16.96351489687606, 19.09832209947511, 17.32320538275583], [5.858585858585858, 17.907021007161138, 19.39194458172433, 17.628155547202617], [5.959595959595959, 18.854333005910238, 19.68556706397356, 17.933105711649404], [6.0606060606060606, 37.70264394397289, 19.979189546222788, 18.23805587609619], [6.161616161616162, 18.299189508184668, 20.272812028472018, 18.543006040542977], [6.262626262626262, 17.681543813872757, 20.56643451072124, 18.84795620498976], [6.363636363636363, 17.89470246682842, 20.86005699297047, 19.152906369436547], [6.4646464646464645, 20.20646521633359, 21.153679475219697, 19.457856533883334], [6.565656565656566, 21.05320972554052, 21.447301957468927, 19.762806698330124], [6.666666666666667, 19.927989878419666, 21.740924439718153, 20.06775686277691], [6.767676767676767, 21.306563200922326, 22.03454692196738, 20.372707027223694], [6.8686868686868685, 20.96769663110824, 22.328169404216606, 20.67765719167048], [6.96969696969697, 20.263971154485787, 22.621791886465832, 20.982607356117267], [7.070707070707071, 41.573516817629624, 22.915414368715062, 21.287557520564054], [7.171717171717171, 23.053188081617485, 23.209036850964285, 21.592507685010837], [7.2727272727272725, 21.782355779071864, 23.502659333213515, 21.897457849457627], [7.373737373737374, 23.685855777026127, 23.79628181546274, 22.202408013904414], [7.474747474747475, 19.80449732015268, 24.08990429771197, 22.5073581783512], [7.575757575757575, 23.54917523164795, 24.383526779961194, 22.812308342797984], [7.6767676767676765, 23.117350098541202, 24.677149262210424, 23.11725850724477], [7.777777777777778, 23.034325982867465, 24.97077174445965, 23.422208671691557], [7.878787878787879, 23.728124412899138, 25.26439422670888, 23.727158836138344], [7.979797979797979, 21.951825024793045, 25.558016708958103, 24.03210900058513], [8.080808080808081, 44.02275235458673, 25.851639191207333, 24.337059165031917], [8.181818181818182, 24.902567116966292, 26.14526167345656, 24.642009329478704], [8.282828282828282, 26.32637889322636, 26.438884155705786, 24.946959493925487], [8.383838383838384, 24.633244933241507, 26.732506637955016, 25.251909658372277], [8.484848484848484, 24.646051851652267, 27.026129120204242, 25.55685982281906], [8.585858585858587, 25.25581871399122, 27.319751602453472, 25.86180998726585], [8.686868686868687, 26.976008178308135, 27.613374084702695, 26.166760151712634], [8.787878787878787, 26.692387473296044, 27.90699656695192, 26.47171031615942], [8.88888888888889, 26.136906462899628, 28.20061904920115, 26.776660480606207], [8.98989898989899, 27.482964402810325, 28.494241531450378, 27.081610645052994], [9.09090909090909, 47.36980482207531, 28.787864013699604, 27.386560809499777], [9.191919191919192, 28.54440256629047, 29.081486495948834, 27.691510973946567], [9.292929292929292, 27.176734784910522, 29.375108978198057, 27.99646113839335], [9.393939393939394, 27.854156035220416, 29.66873146044729, 28.30141130284014], [9.494949494949495, 28.09274033171633, 29.962353942696513, 28.606361467286924], [9.595959595959595, 27.324363839746667, 30.25597642494574, 28.91131163173371], [9.696969696969697, 29.38702936797367, 30.54959890719497, 29.2162617961805], [9.797979797979798, 29.65499466611928, 30.843221389444196, 29.521211960627284], [9.8989898989899, 29.70208315361216, 31.136843871693426, 29.826162125074074],
 [10.0, 29.765412866624853, 31.430466353942652, 30.131112289520857]]);

    // Chart options
    var options = {
      title: 'OLS vs Huber Regression',
      hAxis: { title: 'X', minValue: 0 },
      vAxis: { title: 'y', minValue: 0 },
      pointSize: 2,
      legend: { position: 'right' },
      series: {
        0: { color: 'black', pointShape: 'circle' }, // Observed data points
        1: { color: 'red', lineWidth: 2, pointSize: 0 },          // OLS Regression line
        2: { color: 'green', lineWidth: 2, pointSize: 0 },        // Huber Regression line
      },
    };

    // Render the chart
    var chart = new google.visualization.ScatterChart(document.getElementById('chart_div2'));
    chart.draw(data, options);
  }
</script>
<div id="chart_div2" style="width: 100%; height: 400px;"></div>


---
<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>


