# Problem 10: KNN Regression — Training and Comparison

Using the same Training and Test data from the previous questions, train a `KNeighborsRegressor` model with:

```python
n_neighbors=5
```

Then:

1. Make predictions for the Test data.
2. Calculate MAE, MSE, and RMSE.
3. Compare the results with the Linear Regression model.

Next, set `K` to the following values:

* `K = 1`
* `K = 3`
* `K = 10`
* `K = 20`

For each value of `K`, calculate the MAE.

Then answer the following questions:

1. Which value of `K` produces the lowest MAE?
2. Which value of `K` produces the highest R²?
3. Why might `K = 1` make the model sensitive to noise?
4. Why might a very large value of `K` eliminate local patterns in the data?
5. Is there one value of `K` that is best for every dataset?
6. How is KNN Regression different from Linear Regression in terms of how predictions are made?