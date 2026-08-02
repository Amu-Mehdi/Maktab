# Linear Regression

## Definition

**Linear Regression** is a supervised machine learning algorithm used to model the relationship between one or more independent variables (features) and a continuous dependent variable (target). It assumes that this relationship is **linear**, meaning changes in the input variables produce proportional changes in the output.

The primary purpose of Linear Regression is **prediction** and **inference**:
- **Prediction** — estimating an unknown output value based on known inputs.
- **Inference** — understanding how strongly each feature influences the target.

It is typically used when:
- The target variable is continuous (e.g., price, temperature, salary).
- There is a reasonably linear relationship between inputs and output.
- Interpretability is important alongside predictive performance.

---

## Mathematical Formula

### Simple Linear Regression

Used when there is a single independent variable:

$$
y = \beta_0 + \beta_1x + \varepsilon
$$

**Where:**
- $y$ — the dependent variable (target/output)
- $x$ — the independent variable (feature/input)
- $\beta_0$ — the intercept (value of $y$ when $x = 0$)
- $\beta_1$ — the slope (change in $y$ for a one-unit change in $x$)
- $\varepsilon$ — the error term (captures noise/unexplained variation)

### Multiple Linear Regression

Used when there are multiple independent variables:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \varepsilon
$$

**Where:**
- $x_1, x_2, \dots, x_n$ — independent variables (features)
- $\beta_1, \beta_2, \dots, \beta_n$ — coefficients representing the effect of each feature on $y$
- $\beta_0$ — the intercept term
- $\varepsilon$ — the error term

### Matrix Notation

For efficient computation, the model is expressed in matrix form:

$$
\hat{y} = X\beta
$$

**Where:**
- $\hat{y}$ — vector of predicted values
- $X$ — matrix of input features (including a column of 1s for the intercept)
- $\beta$ — vector of coefficients (parameters) to be learned

### Cost Function (Mean Squared Error)

The model learns by minimizing the **cost function**, which measures the average squared difference between predicted and actual values:

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

**Where:**
- $J(\theta)$ — the cost function to be minimized
- $m$ — number of training examples
- $h_\theta(x^{(i)})$ — the model's predicted value for the $i$-th example
- $y^{(i)}$ — the actual (true) value for the $i$-th example
- $\theta$ — the set of model parameters ($\beta_0, \beta_1, \dots$)
- The factor $\frac{1}{2}$ simplifies the derivative during optimization

### Gradient Descent Update Equations

Gradient Descent iteratively adjusts $\theta$ to minimize $J(\theta)$:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$

**Where:**
- $\theta_j$ — the $j$-th parameter being updated
- $\alpha$ — the learning rate (controls step size)
- $\frac{\partial}{\partial \theta_j}J(\theta)$ — the gradient (partial derivative) of the cost function with respect to $\theta_j$
- $:=$ — denotes an update/assignment operation

### Closed-Form Solution (Normal Equation)

Instead of iterative optimization, coefficients can be computed directly:

$$
\beta = (X^TX)^{-1}X^Ty
$$

**Where:**
- $X^T$ — transpose of the feature matrix $X$
- $(X^TX)^{-1}$ — inverse of the matrix product $X^TX$
- $y$ — vector of actual target values

> **Note:** The Normal Equation is efficient for small-to-medium datasets but becomes computationally expensive when the number of features is very large, since matrix inversion has cubic time complexity.

---

## Intuition

- **Best-fit line:** Linear Regression tries to draw a straight line (or hyperplane in higher dimensions) that best represents the trend in the data.
- **Relationship between variables:** It quantifies how much the target changes as each feature changes.
- **Error minimization:** The "best" line is the one that minimizes the total squared distance between actual data points and predicted values (residuals).
- **Prediction concept:** Once the line is fitted, any new input can be plugged into the equation to generate a predicted output.

Think of it as finding the line that passes as close as possible to all data points simultaneously — balancing overall error rather than fitting any single point perfectly.

---

## Example

### Sample Dataset

| House Size (sq. ft) | Price (in thousand USD) |
|---------------------|-----------------|
| 1000                | 200             |
| 1500                | 300             |
| 2000                | 400             |
| 2500                | 500             |

### Manual Prediction Example

From the pattern above, we can estimate:

$$
\text{Price} = 0 + 0.2 \times \text{Size}
$$

For a house of **1800 sq. ft**:

$$
\text{Price} = 0.2 \times 1800 = 360 \text{ (thousand USD)}
$$

So the predicted price is approximately **\$360,000**.

### Python Implementation (scikit-learn)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Feature (X) and target (y)
X = np.array([[1000], [1500], [2000], [2500]])
y = np.array([200, 300, 400, 500])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a 1800 sq. ft house
prediction = model.predict([[1800]])

print("Intercept (β0):", model.intercept_)
print("Coefficient (β1):", model.coef_)
print("Predicted Price:", prediction)
```

### Interpretation of Output

- `Intercept (β0)` — the baseline price when size is 0 (often not meaningful on its own, but mathematically necessary).
- `Coefficient (β1)` — how much the price increases for each additional square foot.
- `Predicted Price` — the model's estimated price for the given input size.

---

## Advantages

- **Simplicity** — easy to understand, implement, and explain.
- **Interpretability** — coefficients directly show feature impact on the target.
- **Fast training** — computationally inexpensive, even on large datasets.
- **Works well for linear relationships** — highly effective when the true relationship is approximately linear.
- **Strong baseline model** — often used as a benchmark before trying more complex algorithms.

---

## Limitations

- **Assumes linearity** — fails to capture complex, nonlinear relationships between variables.
- **Sensitive to outliers** — extreme values can heavily skew the fitted line.
- **Multicollinearity** — highly correlated features can distort coefficient estimates and reduce interpretability.
- **Underfitting nonlinear relationships** — performs poorly when data follows a curved or complex pattern.
- **Assumptions about residuals** — requires errors to be normally distributed, independent, and have constant variance (homoscedasticity); violations reduce model reliability.

---

## Applications

- **House price prediction** — estimating property values based on size, location, and amenities.
- **Sales forecasting** — predicting future sales based on historical trends and marketing spend.
- **Healthcare** — modeling relationships such as dosage vs. patient response.
- **Finance** — predicting stock trends, risk scores, or loan default probabilities.
- **Economics** — analyzing relationships between GDP, inflation, and employment.
- **Marketing** — measuring the impact of advertising spend on revenue.
- **Risk analysis** — quantifying relationships between risk factors and outcomes.

---

## Related Concepts

- **Multiple Linear Regression** — extends simple regression to multiple input features.
- **Polynomial Regression** — models nonlinear relationships by adding polynomial terms of features.
- **Ridge Regression** — adds L2 regularization to reduce overfitting and handle multicollinearity.
- **Lasso Regression** — adds L1 regularization, capable of shrinking some coefficients to zero (feature selection).
- **Elastic Net** — combines L1 and L2 regularization for balanced control.
- **Gradient Descent** — an iterative optimization algorithm used to minimize the cost function.
- **Mean Squared Error (MSE)** — a common metric measuring average squared prediction error.
- **R² Score** — indicates the proportion of variance in the target explained by the model (closer to 1 is better).
- **Adjusted R²** — a modified R² that accounts for the number of predictors, penalizing unnecessary complexity.
- **Feature Engineering** — the process of creating or transforming features to improve model performance.
- **Regularization** — techniques that constrain model complexity to prevent overfitting.

---

## References

- [Scikit-learn Documentation — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- *An Introduction to Statistical Learning (ISLR)* — [https://www.statlearning.com/](https://www.statlearning.com/)
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron — [O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Andrew Ng's Machine Learning Course (Coursera)](https://www.coursera.org/learn/machine-learning)
- [Wikipedia — Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
