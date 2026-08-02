# Mean Squared Error (MSE)

## Definition

**Mean Squared Error (MSE)** is one of the most widely used metrics for evaluating the performance of regression models. It measures the **average of the squared differences** between the actual (true) values and the predicted values.

MSE tells us, on average, how far the model's predictions deviate from the real data — with larger errors being penalized much more heavily than smaller ones (because the differences are squared).

It is typically used when:
- Evaluating or comparing regression models.
- Serving as the **cost function** during model training (e.g., in Linear Regression).
- Large errors are considered especially undesirable and should be penalized strongly.

---

## Mathematical Formula

### MSE Formula

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

**Where:**
- $MSE$ — the Mean Squared Error value
- $n$ — the total number of data points (observations)
- $y_i$ — the actual (true) value of the $i$-th observation
- $\hat{y}_i$ — the predicted value of the $i$-th observation
- $\sum_{i=1}^{n}$ — summation over all $n$ observations
- $(y_i - \hat{y}_i)$ — the residual (error) for the $i$-th observation
- $(y_i - \hat{y}_i)^2$ — the squared residual, ensuring all errors are positive and larger errors are penalized more

### MSE as a Cost Function (used in training)

When used as a training objective (e.g., for Linear Regression), MSE is often written as:

$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
$$

**Where:**
- $J(\theta)$ — the cost function to be minimized
- $m$ — number of training examples
- $h_\theta(x^{(i)})$ — the model's predicted value for the $i$-th example
- $y^{(i)}$ — the actual value for the $i$-th example
- $\theta$ — the model's parameters (weights)
- The factor $\frac{1}{2}$ is included purely for mathematical convenience — it cancels out nicely when taking the derivative during Gradient Descent

### Root Mean Squared Error (RMSE)

A closely related metric that converts MSE back to the same unit as the target variable:

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} = \sqrt{MSE}
$$

**Where:**
- $RMSE$ — Root Mean Squared Error
- All other symbols are the same as defined above

---

## Intuition

- **Measuring "wrongness":** MSE quantifies how wrong the model's predictions are, on average, across all data points.
- **Squaring the errors:** Squaring removes negative signs (so overestimates and underestimates don't cancel each other out) and **amplifies large errors** more than small ones.
- **Penalizing big mistakes:** A prediction that is off by 10 contributes 100 to the error, while one off by 2 contributes only 4 — meaning MSE is very sensitive to large deviations and outliers.
- **Lower is better:** An MSE of 0 means perfect predictions; the higher the MSE, the worse the model's predictions.
- **Not in the original unit:** Because errors are squared, MSE's unit is the *square* of the target variable's unit (e.g., if predicting price in dollars, MSE is in dollars²) — this is why RMSE is often preferred for easier interpretation.

---

## Example

### Sample Dataset

| Actual Price ($y$) | Predicted Price ($\hat{y}$) | Error ($y - \hat{y}$) | Squared Error |
|---------------------|------------------------------|--------------------------|----------------|
| 200                  | 210                          | -10                       | 100            |
| 300                  | 290                          | 10                        | 100            |
| 400                  | 420                          | -20                       | 400            |
| 500                  | 480                          | 20                        | 400            |

### Manual Calculation

$$
MSE = \frac{100 + 100 + 400 + 400}{4} = \frac{1000}{4} = 250
$$

So the **Mean Squared Error is 250** (in squared thousand-USD units, based on the earlier dataset).

To interpret this in the original scale, we can compute RMSE:

$$
RMSE = \sqrt{250} \approx 15.81
$$

This means predictions are, on average, off by about \$15,810 (given the dataset was measured in thousands of USD).

### Python Implementation (scikit-learn)

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# Actual and predicted values
y_actual = np.array([200, 300, 400, 500])
y_predicted = np.array([210, 290, 420, 480])

# Calculate MSE
mse = mean_squared_error(y_actual, y_predicted)

# Calculate RMSE
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
```

### Interpretation of Output

- `MSE` — the average squared error across all predictions; useful for comparing models but hard to interpret directly due to squared units.
- `RMSE` — the same error expressed in the original unit of the target variable, making it easier to explain (e.g., "predictions are off by ~15.81 on average").
- A **lower MSE/RMSE** indicates a better-fitting model; a **higher MSE/RMSE** indicates the model's predictions deviate more from actual values.

---

## Advantages

- **Differentiable** — smooth and easy to optimize using Gradient Descent, making it ideal as a training loss function.
- **Penalizes large errors** — strongly discourages predictions that are far off, which is useful when big mistakes are especially costly.
- **Simple and widely understood** — one of the most common and well-established regression metrics.
- **Mathematically convenient** — has a unique minimum for linear models, making optimization straightforward.

---

## Limitations

- **Sensitive to outliers** — a few large errors can dominate the metric and misrepresent overall model performance.
- **Not in original units** — MSE's squared units make it less intuitive to interpret directly (RMSE is often used instead).
- **Scale-dependent** — MSE values are not comparable across datasets with different target scales (e.g., comparing house prices vs. stock prices).
- **Doesn't indicate direction of error** — MSE shows *how much* error exists, but not whether the model tends to overpredict or underpredict.

---

## Related Metrics

- **Root Mean Squared Error (RMSE)** — the square root of MSE; expressed in the same unit as the target variable.
- **Mean Absolute Error (MAE)** — averages the absolute (not squared) errors; less sensitive to outliers than MSE.
- **R² Score (Coefficient of Determination)** — measures the proportion of variance in the target explained by the model.
- **Adjusted R²** — a variation of R² that penalizes unnecessary model complexity.
- **Mean Absolute Percentage Error (MAPE)** — expresses error as a percentage, useful for comparing across different scales.
- **Huber Loss** — a hybrid loss function that behaves like MSE for small errors and like MAE for large errors, reducing outlier sensitivity.

### Quick Comparison

| Metric | Sensitive to Outliers? | Same Unit as Target? | Common Use |
|--------|--------------------------|------------------------|-------------|
| MSE    | Yes (highly)              | No (squared)            | Training loss, model comparison |
| RMSE   | Yes                       | Yes                     | Interpretable error reporting |
| MAE    | No (less)                 | Yes                     | Robust error reporting |
| R²     | Moderate                 | N/A (ratio)             | Explained variance |

---

## Applications

- **Model training** — used as the cost function in Linear Regression and many other regression algorithms.
- **Model evaluation** — comparing different models (e.g., Linear Regression vs. Random Forest) to select the best-performing one.
- **Hyperparameter tuning** — used as the metric to optimize during cross-validation and grid search.
- **Forecasting accuracy** — assessing how well sales, demand, or financial forecasts match actual outcomes.
- **Anomaly detection** — large squared errors can help flag unusual or unexpected data points.

---

## References

- [Scikit-learn Documentation — Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- *An Introduction to Statistical Learning (ISLR)* — [https://www.statlearning.com/](https://www.statlearning.com/)
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron — [O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Andrew Ng's Machine Learning Course (Coursera)](https://www.coursera.org/learn/machine-learning)
- [Wikipedia — Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)
