# Problem 11: Logistic Regression — Predicting Student Pass/Fail

Before training the model, split the data into Training and Test sets using:

```python
test_size=0.2
random_state=42
```

Train the model only using the Training data.

Then make both class predictions and class-probability predictions for the Test data.

To convert the `Passed` column into numerical labels, use:

* `Yes` → `1`
* `No` → `0`

Select `StudyHours` as the Feature and train a `LogisticRegression` model.

Then:

1. Predict the classes using `predict()`.
2. Obtain the class probabilities using `predict_proba()`.

Based on the outputs, answer the following questions:

1. Why is this a Classification problem?
2. What does each output of `predict()` represent?
3. What do the two columns returned by `predict_proba()` represent?
4. If the probability of class `1` is `0.72`, what is the final class using a threshold of `0.5`?
5. If the probability of class `1` is `0.48`, what is the final class?
6. Why is Linear Regression not suitable for predicting probabilities?
7. Why is Logistic Regression a classification algorithm despite having "Regression" in its name?