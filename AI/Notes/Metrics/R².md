# R² Score (Coefficient of Determination)

## Definition

The **R² Score**, also known as the **Coefficient of Determination**, is one of the most widely used evaluation metrics for regression models.

It measures **how well a regression model explains the variability of the target variable**.

Unlike error-based metrics such as MAE or RMSE, which measure the magnitude of prediction errors, **R² evaluates how much of the variation in the observed data is captured by the model.**

In simple terms:

> **R² answers the question: "How much of the information contained in the data has the model successfully learned?"**

---

## Why This Metric Exists

Suppose you build a regression model to predict house prices.

After training, someone asks:

> "Is this model actually good?"

Metrics like MAE and RMSE tell us **how far the predictions are from the actual values**, but they do **not** tell us whether the model has learned the underlying pattern in the data.

For example:

Model A

```
Average Error = $20,000
```

Model B

```
Average Error = $25,000
```

Is Model A significantly better?

Maybe.

Maybe not.

The answer depends on how much variation exists in the dataset.

If house prices range from:

```
$100,000
to
$10,000,000
```

then a $25,000 error is relatively small.

But if prices range from:

```
$100,000
to
$130,000
```

the same error is enormous.

Therefore, we need a metric that evaluates prediction errors **relative to the variability of the data**.

This is exactly why R² was introduced.

---

## Intuition

Imagine three students taking an exam.

### Student A

Always answers:

```
50
```

regardless of the question.

---

### Student B

Makes predictions that roughly follow the trend.

```
Question Difficulty ↑

Predicted Score ↑
```

---

### Student C

Almost perfectly predicts every score.

Which student actually understands the relationship between question difficulty and score?

Obviously:

```
Student C
```

R² tries to answer this exact question.

It measures how much of the observed variation is explained by the model.

---

### Another Analogy

Imagine throwing darts.

```
        🎯
```

The center represents the actual values.

If your darts are randomly scattered,

your model explains almost nothing.

If your darts cluster around the center,

your model explains most of the variation.

The tighter the cluster,

the higher the R² score.

---

### House Price Example

Suppose five houses have prices:

| House | Price |
|-------|-------:|
| A | 100 |
| B | 150 |
| C | 200 |
| D | 250 |
| E | 300 |

Notice that prices vary significantly.

Now compare two models.

Model 1 predicts:

| Prediction |
|-----------:|
|102|
|149|
|198|
|251|
|299|

This model captures almost all variability.

Its R² will be close to **1**.

---

Model 2 predicts:

| Prediction |
|-----------:|
|200|
|200|
|200|
|200|
|200|

It ignores all variation.

It simply predicts the average.

Its R² will be approximately **0**.

---

### The Key Intuition

R² does **not** ask:

> "How wrong are the predictions?"

Instead, it asks:

> **"How much of the data's variability has the model explained?"**

This single idea is the foundation of R².

---

# Mathematical Formula

## Formula

\[
R^2
=
1-
\frac{SS_{res}}
{SS_{tot}}
\]

---

## Formula Breakdown

### 1. Residual Sum of Squares (SSres)

\[
SS_{res}
=
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
\]

This represents the **unexplained variation**.

It measures how much error remains after the model makes predictions.

Where:

- \(y_i\) = Actual value
- \(\hat y_i\) = Predicted value
- \(n\) = Number of observations
- \(\Sigma\) = Summation over all samples

The closer predictions are to actual values,

the smaller \(SS_{res}\) becomes.

---

### 2. Total Sum of Squares (SStot)

\[
SS_{tot}
=
\sum_{i=1}^{n}
(y_i-\bar y)^2
\]

This represents the **total variation** in the dataset.

Where:

- \(\bar y\) = Mean of the target values

It answers the question:

> "How much do the actual values vary around their average?"

---

### 3. The Mean

\[
\bar y
=
\frac{1}{n}
\sum_{i=1}^{n}
y_i
\]

The mean acts as a **baseline predictor**.

If a model predicts only the mean,

then

\[
SS_{res}=SS_{tot}
\]

which results in

\[
R^2=0
\]

---

## Why Does It Work This Way?

The formula

\[
R^2
=
1-
\frac{SS_{res}}
{SS_{tot}}
\]

compares two quantities.

### Total Information

```
SS_tot
```

represents

```
Everything that exists
inside the data
```

---

### Remaining Error

```
SS_res
```

represents

```
Everything the model
failed to explain
```

---

The ratio

\[
\frac{SS_{res}}
{SS_{tot}}
\]

tells us:

> "What fraction of the original variability remains unexplained?"

Finally,

subtracting this ratio from 1 gives

the explained proportion.

---

### Why Use Squared Errors?

Both SSres and SStot use squared differences.

Reasons:

- Prevent positive and negative errors from canceling each other.
- Penalize large errors more heavily.
- Produce smooth mathematical properties useful in statistics.

---

### Why Compare Against the Mean?

Without a baseline,

we cannot determine whether a model is actually useful.

The mean prediction is the simplest possible regression model.

R² answers:

> "How much better is the model than simply predicting the average?"

---

# Cost Function (If Applicable)

Strictly speaking,

**R² is NOT a cost function.**

Instead,

it is an **evaluation metric**.

During model training,

algorithms typically optimize:

- Mean Squared Error (MSE)
- Huber Loss
- MAE
- Other differentiable loss functions

After training,

R² is computed to evaluate overall model quality.

---

# Manual Calculation

## Sample Dataset

| Sample | Actual | Predicted |
|--------|-------:|----------:|
|1|3|2.5|
|2|5|5.2|
|3|7|6.8|
|4|9|8.5|
|5|11|10.7|

---

## Step 1 — Calculate Errors

|Actual|Prediction|Error|
|------:|---------:|----:|
|3|2.5|0.5|
|5|5.2|-0.2|
|7|6.8|0.2|
|9|8.5|0.5|
|11|10.7|0.3|

---

## Step 2 — Square the Errors

|Error|Squared Error|
|----:|------------:|
|0.5|0.25|
|-0.2|0.04|
|0.2|0.04|
|0.5|0.25|
|0.3|0.09|

Therefore,

\[
SS_{res}
=
0.25+0.04+0.04+0.25+0.09
=
0.67
\]

---

## Step 3 — Calculate SS_tot

Mean:

\[
\bar y
=
\frac{3+5+7+9+11}{5}
=
7
\]

Now compute deviations from the mean.

|Actual|Mean|Difference|Squared|
|------:|---:|---------:|-------:|
|3|7|-4|16|
|5|7|-2|4|
|7|7|0|0|
|9|7|2|4|
|11|7|4|16|

Therefore,

\[
SS_{tot}
=
16+4+0+4+16
=
40
\]

---

## Step 4 — Compute the Final Metric

\[
R^2
=
1-
\frac{0.67}{40}
\]

\[
=
1-0.01675
\]

\[
=
0.98325
\]

---

## Final Result

\[
\boxed{
R^2
=
0.983
}
\]

Interpretation:

The model explains approximately **98.3% of the variance** in the target variable.

This indicates an excellent fit.

---

# Python Implementation

## scikit-learn Example

```python
from sklearn.metrics import r2_score

y_true = [3, 5, 7, 9, 11]
y_pred = [2.5, 5.2, 6.8, 8.5, 10.7]

score = r2_score(y_true, y_pred)

print(score)
```

Output

```python
0.98325
```

---

## Interpretation of the Output

```
R² = 0.983
```

means

> The regression model explains approximately **98.3% of the variability** in the observed target values.

Notice that this **does not mean**:

- The model is correct 98.3% of the time.
- Predictions are 98.3% accurate.

Instead,

it means

the model captures almost all of the underlying variation present in the data.

---

# Interpretation

### Lower Value

A low R² indicates that the model explains only a small portion of the variability in the target variable.

The model has failed to capture the underlying relationship effectively.

---

### Higher Value

A higher R² indicates that the model explains a larger portion of the observed variation.

Generally,

higher is better.

---

### Best Possible Value

\[
R^2=1
\]

This means

```
Perfect predictions.
```

Every predicted value equals its corresponding actual value.

---

### Value Range

Common values are

\[
0
\le
R^2
\le
1
\]

However,

R² can also become **negative**.

Negative values indicate that

the model performs **worse than simply predicting the mean**.

---

### Units

R² has **no physical unit**.

It is a dimensionless metric because it is a ratio of two quantities with identical units.

---

### How to Interpret the Metric

| R² | Interpretation |
|----:|----------------|
|1.0|Perfect fit|
|0.9–1.0|Excellent|
|0.7–0.9|Good|
|0.5–0.7|Moderate|
|0–0.5|Weak|
|0|Equivalent to predicting the mean|
|<0|Worse than predicting the mean|


# Behavior Analysis

Understanding **how R² behaves under different prediction scenarios** is far more important than simply memorizing its formula.

Recall the definition:

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

Everything about R² can be understood by observing the relationship between:

- **SSres** → Unexplained variation (prediction errors)
- **SStot** → Total variation present in the dataset

Whenever **SSres decreases**, R² increases.

Whenever **SSres increases**, R² decreases.

---

## Case 1 — Many Small Errors

Suppose a regression model predicts house prices.

| Actual | Prediction |
|--------:|-----------:|
|100|102|
|200|198|
|300|305|
|400|399|
|500|503|

Errors are small and evenly distributed.

```
Error Distribution

+2
-2
+5
-1
+3
```

Since every prediction is close to the actual value,

the residual sum of squares remains small.

Therefore,

```
Small SSres
↓

Large R²
```

### Interpretation

The model successfully captures almost all of the underlying structure in the data.

---

## Case 2 — One Large Error

Now consider another model.

| Actual | Prediction |
|--------:|-----------:|
|100|102|
|200|198|
|300|305|
|400|399|
|500|2000|

Four predictions are excellent.

One prediction is catastrophically wrong.

The last error is

```
1500
```

After squaring:

\[
1500^2 = 2,250,000
\]

This single observation dominates the entire residual sum of squares.

Consequently,

```
Huge SSres
↓

Much Lower R²
```

### Interpretation

Although most predictions are good,

one extremely poor prediction significantly reduces the overall model quality.

---

## Case 3 — Uniform Errors

Imagine every prediction has approximately the same error.

```
10
10
10
10
10
```

Will R² always be identical?

No.

The answer depends on the variability of the dataset.

---

### Dataset A

```
100
200
300
400
500
```

The target values vary considerably.

A constant error of 10 is relatively small.

Therefore,

R² remains high.

---

### Dataset B

```
100
101
102
103
104
```

The exact same prediction error now represents a large proportion of the total variation.

Consequently,

R² becomes much lower.

---

### Key Insight

R² evaluates errors **relative to the natural variability of the dataset**.

It is **not an absolute error metric**.

---

## Case 4 — High Variance Dataset

Suppose actual values span a very large range.

```
100
500
900
1500
2500
```

Even moderate prediction errors may still produce a high R² because

```
SStot
```

is very large.

---

## Case 5 — Low Variance Dataset

Now suppose the data hardly changes.

```
100
101
100
99
100
```

Even small prediction errors may lead to a poor R² because

```
SStot
```

is extremely small.

---

## Case 6 — Predicting the Mean

Suppose the model predicts

```
30
30
30
30
30
```

for every observation.

This is equivalent to predicting the average.

In this situation

\[
SS_{res}=SS_{tot}
\]

Therefore

\[
R^2=0
\]

Meaning

> The model has learned nothing beyond the average.

---

## Case 7 — Worse Than Predicting the Mean

Suppose predictions are

```
100
100
100
100
100
```

while actual values are

```
10
20
30
40
50
```

Now

\[
SS_{res}>SS_{tot}
\]

which gives

\[
R^2<0
\]

This means

> The model is worse than the simplest possible baseline.

---

## Visual Intuition

```
Perfect Model

Actual
    ●
Pred
    ●

R² = 1
```

---

```
Good Model

Actual
● ● ● ● ●

Pred
 ● ● ● ● ●

R² ≈ High
```

---

```
Average Predictor

Actual
● ● ● ● ●

Pred
──────────

R² = 0
```

---

```
Terrible Predictor

Actual
● ● ● ● ●

Pred
        ●●●●●

R² < 0
```

---

# Advantages

## 1. Easy to Interpret

R² provides a clear interpretation.

```
R² = 0.90
```

means

> The model explains approximately 90% of the observed variability.

---

## 2. Dimensionless

Unlike MAE or RMSE,

R² has no units.

Therefore,

it can be compared across problems more easily.

---

## 3. Excellent for Comparing Models

Suppose three regression models produce

| Model | R² |
|------|----:|
|Linear Regression|0.81|
|Random Forest|0.92|
|XGBoost|0.95|

R² immediately indicates which model explains more variation.

---

## 4. Provides a Baseline

Unlike MAE,

R² automatically compares your model against

```
Predicting the mean.
```

This makes interpretation much easier.

---

## 5. Widely Used

R² is one of the most recognized regression metrics in

- Machine Learning
- Statistics
- Economics
- Scientific Research

---

## 6. Excellent for Measuring Overall Model Quality

Rather than focusing only on prediction errors,

R² evaluates how well the model captures the underlying data distribution.

---

## 7. Complements Error Metrics

R² works particularly well alongside

- MAE
- RMSE

Together they provide a comprehensive evaluation.

---

# Limitations

## 1. Does Not Measure Actual Prediction Error

A model may have

```
R² = 0.95
```

while still producing very large prediction errors.

Always inspect MAE or RMSE as well.

---

## 2. Sensitive to Outliers

Because

\[
SS_{res}
\]

uses squared errors,

one extremely poor prediction can drastically reduce R².

---

## 3. High R² Does Not Guarantee a Good Model

A highly overfitted model may produce

```
Training R² = 0.99

Testing R² = 0.42
```

Always evaluate on unseen data.

---

## 4. Depends on Dataset Variance

The same prediction error can produce

very different R² values

depending on

```
SStot
```

---

## 5. Cannot Measure Causality

A high R² only indicates

strong statistical fit.

It does not imply

cause-and-effect relationships.

---

## 6. May Increase When Adding Features

Adding more input variables often increases R²,

even when those variables contribute little useful information.

This is why

**Adjusted R²**

was introduced.

---

## 7. Not Suitable as a Training Loss

Most optimization algorithms minimize

- MSE
- MAE
- Huber Loss

rather than maximizing R² directly.

---

## 8. Difficult to Compare Across Different Problems

An R² of

```
0.80
```

in stock prediction

may be extraordinary,

while the same value

in manufacturing

may be unacceptable.

---

# When Should You Use This Metric?

Use R² when

✅ Comparing multiple regression models

✅ Measuring overall model quality

✅ Evaluating how much variance the model explains

✅ Writing scientific papers

✅ Benchmarking regression algorithms

✅ Reporting regression performance

---

# When Should You Avoid This Metric?

Avoid relying solely on R² when

❌ Actual prediction error matters

❌ Large prediction errors are critical

❌ Your dataset contains many outliers

❌ Training neural networks

❌ Comparing unrelated datasets

---

# Practical Rule of Thumb

Use the following guideline:

```
Need average prediction error?

↓

Use MAE
```

---

```
Need to heavily penalize large errors?

↓

Use RMSE
```

---

```
Need overall model quality?

↓

Use R²
```

---

```
Best practice?

↓

Use MAE + RMSE + R² together.
```

---

# Common Misconceptions

## ❌ Misconception 1

Higher R² always means a better model.

### Reality

Not necessarily.

A model with

```
R² = 0.95
```

may still have unacceptable prediction errors.

---

## ❌ Misconception 2

R² is always between 0 and 1.

### Reality

False.

R² may become negative.

Negative values indicate

performance worse than predicting the mean.

---

## ❌ Misconception 3

R² measures prediction accuracy.

### Reality

No.

It measures

explained variance,

not prediction accuracy.

---

## ❌ Misconception 4

R² alone is sufficient.

### Reality

Professional machine learning engineers almost never report R² by itself.

---

## ❌ Misconception 5

A low R² always indicates a poor model.

### Reality

Not necessarily.

Fields like

- Finance
- Economics
- Human behavior

naturally exhibit much lower R² values.

---

## ❌ Misconception 6

R² can compare every regression problem.

### Reality

Only compare R² values

on the same prediction task.

---

# Comparison with Other Regression Metrics

| Feature | R² Score | MAE | MSE | RMSE |
|----------|-----------|------|------|-------|
|Measures average error|❌|✅|❌|✅|
|Measures explained variance|✅|❌|❌|❌|
|Uses squared errors|Indirectly|❌|✅|✅|
|Sensitive to outliers|High|Low|Very High|High|
|Has physical units|No|Yes|Squared Units|Yes|
|Easy for business interpretation|Moderate|Excellent|Poor|Good|
|Useful for optimization|No|Sometimes|Yes|Sometimes|
|Best for model comparison|Excellent|Moderate|Moderate|Good|
|Can become negative|Yes|No|No|No|
|Ideal Value|1|0|0|0|

---

## Summary

R² is one of the most informative metrics for evaluating regression models because it answers a unique question:

> **"How much of the variability in the target variable has the model successfully explained?"**

However,

R² should almost never be used in isolation.

A complete regression evaluation typically reports

- MAE
- RMSE
- R²

together,

providing both **absolute error information** and **overall model quality**.



# Practical Examples

## House Price Prediction

Suppose you build a regression model to estimate house prices using features such as:

- Area
- Number of bedrooms
- Age of the building
- Neighborhood
- Distance to the city center

Evaluation results:

| Metric | Value |
|---------|------:|
| MAE | \$18,000 |
| RMSE | \$27,000 |
| R² | 0.91 |

### Interpretation

- The model explains **91% of the variation** in house prices.
- On average, predictions are **\$18,000 away** from the actual prices.
- Some larger prediction errors still exist (RMSE > MAE).

This is a typical example where all three metrics should be reported together.

---

## Sales Forecasting

Suppose a company predicts weekly sales.

Results:

```
MAE = 250 Units

RMSE = 430 Units

R² = 0.88
```

Interpretation:

- The model captures most sales patterns.
- Average prediction error is 250 units.
- Some weeks contain larger forecasting errors.

---

## Time Series Forecasting

Examples:

- Electricity demand
- Website traffic
- Monthly revenue
- Weather forecasting

Although R² can be useful,

time series models are usually evaluated together with

- MAE
- RMSE
- MAPE

because trends and seasonality may make R² misleading.

---

## Healthcare

Example:

Predicting blood glucose levels.

Requirements:

- Accurate predictions
- Few catastrophic errors

Recommended metrics:

✅ MAE

✅ RMSE

✅ R²

R² alone is insufficient because one large prediction error may have serious medical consequences.

---

## Finance

Examples:

- Stock price prediction
- Portfolio returns
- Risk estimation

Financial data contains considerable randomness.

Therefore,

even

```
R² = 0.35
```

may represent a useful model.

Always interpret R² within the context of the problem.

---

## Deep Learning

Neural networks are usually trained using

- MSE
- MAE
- Huber Loss

After training,

R² is commonly reported as an evaluation metric rather than an optimization objective.

---

# Best Practices

## 1. Never Report R² Alone

Professional reports typically include

```
MAE

RMSE

R²
```

instead of only one metric.

---

## 2. Evaluate on Test Data

Always calculate R² on

- Validation Set
- Test Set

rather than relying only on training performance.

---

## 3. Compare Models on the Same Dataset

Never compare

```
R² = 0.92
```

from one dataset with

```
R² = 0.92
```

from another unrelated dataset.

---

## 4. Watch for Overfitting

Example:

```
Training R² = 0.99

Testing R² = 0.61
```

This strongly suggests overfitting.

---

## 5. Consider the Problem Domain

There is no universal threshold for a "good" R².

Examples:

Manufacturing

```
0.98
```

may be expected.

Finance

```
0.45
```

may already be excellent.

---

## 6. Understand the Baseline

Remember:

```
R² = 0
```

means

Predicting the average performs just as well.

---

# Interview Questions

## Question 1

**What does R² measure?**

### Answer

R² measures the proportion of variance in the target variable explained by the regression model.

---

## Question 2

**Can R² be negative?**

### Answer

Yes.

Negative R² indicates that the model performs worse than simply predicting the mean.

---

## Question 3

**What is the best possible R² value?**

### Answer

```
R² = 1
```

which indicates perfect predictions.

---

## Question 4

**What does R² = 0 mean?**

### Answer

The model performs no better than predicting the average target value.

---

## Question 5

**Why does R² compare against the mean?**

### Answer

The mean prediction serves as the simplest possible baseline.

Without a baseline,

we cannot determine whether the model actually learned anything useful.

---

## Question 6

**Is a higher R² always better?**

### Answer

Not necessarily.

A higher R² may result from overfitting or may still correspond to unacceptable prediction errors.

---

## Question 7

**Why should R² be reported together with MAE or RMSE?**

### Answer

Because R² measures explained variance,

whereas MAE and RMSE measure actual prediction errors.

---

## Question 8

**Which metric is more interpretable for business users?**

### Answer

Usually MAE,

because it is expressed in the original unit of measurement.

---

## Question 9

**Why is R² dimensionless?**

### Answer

Because it is computed as the ratio of two quantities with identical units.

---

## Question 10

**When should Adjusted R² be preferred?**

### Answer

When comparing regression models with different numbers of input features.

Adjusted R² penalizes unnecessary variables.

---

# Frequently Asked Questions (FAQ)

## Is R² a loss function?

No.

It is an evaluation metric.

---

## Is R² suitable for classification?

No.

It is designed exclusively for regression problems.

---

## Can two models have the same R² but different MAE?

Yes.

R² measures explained variance,

while MAE measures average prediction error.

---

## Does R² tell us whether a model is unbiased?

No.

It only measures explained variance.

---

## Should I maximize R² during training?

Typically no.

Training algorithms usually optimize MSE or another differentiable loss function.

---

## Is R² enough for model evaluation?

No.

It should be combined with error metrics.

---

# Key Takeaways

- R² measures **explained variance**, not prediction error.
- The best possible value is **1**.
- R² can become **negative**.
- Higher R² generally indicates a better fit.
- R² compares your model against predicting the mean.
- It is dimensionless.
- It should almost always be reported together with MAE and RMSE.
- High R² does not guarantee a good model.
- Always evaluate R² on unseen data.
- Interpret R² within the context of the application.

---

# Memory Tricks

## Mental Model

Imagine the dataset as a puzzle.

```
Entire Puzzle
↓

SS_tot
```

Your model solves part of the puzzle.

```
Solved Part

↓

Explained Variance
```

The remaining unsolved pieces are

```
SS_res
```

The larger the solved portion,

the larger the R².

---

## Analogy

Imagine a teacher explaining mathematics.

The total amount students need to learn is

```
SS_tot
```

Everything students still do not understand is

```
SS_res
```

The better the teacher,

the less unexplained knowledge remains.

That percentage of explained knowledge is analogous to R².

---

## One Sentence to Remember

> **R² does not ask "How wrong is the model?" It asks "How much of the data has the model successfully explained?"**

---

# Final Summary

The **Coefficient of Determination (R² Score)** is one of the most important evaluation metrics for regression.

Unlike MAE, MSE, and RMSE, which quantify prediction error,

R² evaluates the proportion of variability in the target variable explained by the regression model.

Mathematically,

\[
R^2
=
1-
\frac{SS_{res}}
{SS_{tot}}
\]

where

- **SSres** measures unexplained variation.
- **SStot** measures total variation.

A perfect model produces

```
R² = 1
```

A model equivalent to predicting the average produces

```
R² = 0
```

A model worse than the average predictor produces

```
R² < 0
```

R² is especially useful for

- comparing regression models,
- measuring overall model quality,
- reporting scientific results.

However,

it should **never be interpreted as the average prediction error** and should almost always be used together with

- MAE
- RMSE

to obtain a complete understanding of regression performance.

---

# References

1. Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical Learning*. Springer.

2. James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning*. Springer.

3. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer.

4. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. O'Reilly.

5. Scikit-learn Documentation:
https://scikit-learn.org/stable/modules/model_evaluation.html

6. Montgomery, D. C., Peck, E. A., & Vining, G. G.
*Introduction to Linear Regression Analysis.*

7. Wikipedia – Coefficient of Determination:
https://en.wikipedia.org/wiki/Coefficient_of_determination

8. Murphy, K. P.
*Machine Learning: A Probabilistic Perspective.*

9. ESL and ISLR official companion resources.

10. Scikit-learn API Reference – `r2_score`.