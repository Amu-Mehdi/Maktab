
## Question 2 – Linear Regression for Predicting Online Quiz Scores

An online education platform wants to predict students' final quiz scores based on their learning behavior during a course.

The dataset contains **60 students** and four columns.

```python
import pandas as pd

quiz_data = pd.DataFrame({
    "study_hours": [
        4.8, 11.4, 8.9, 7.4, 2.3, 2.3, 1.2, 10.5, 7.4, 8.6,
        0.7, 11.7, 10.1, 2.9, 2.6, 2.6, 4.0, 6.5, 5.5, 3.8,
        7.5, 2.1, 3.9, 4.7, 5.7, 9.5, 2.8, 6.4, 7.3, 1.0,
        7.5, 2.5, 1.2, 11.4, 11.6, 9.8, 4.0, 1.6, 8.4, 5.6,
        1.9, 6.2, 0.9, 11.0, 3.5, 8.1, 4.1, 6.5, 6.8, 2.6,
        11.7, 9.4, 11.3, 10.8, 7.4, 11.1, 1.5, 2.8, 1.0, 4.2
    ],
    "videos_watched": [
        15, 12, 17, 14, 20, 23, 24, 12, 8, 14,
        12, 0, 24, 6, 8, 23, 0, 11, 7, 23,
        10, 18, 16, 7, 2, 2, 0, 4, 9, 6,
        8, 6, 8, 7, 11, 1, 0, 15, 22, 22,
        23, 4, 2, 11, 7, 21, 2, 0, 2, 4,
        14, 13, 2, 0, 4, 22, 13, 6, 8, 14
    ],
    "practice_solved": [
        14, 25, 12, 31, 38, 31, 3, 29, 36, 22,
        38, 14, 28, 35, 12, 31, 6, 21, 27, 1,
        5, 27, 27, 19, 29, 10, 27, 24, 38, 32,
        0, 26, 12, 2, 38, 5, 7, 26, 8, 36,
        32, 23, 14, 31, 31, 23, 11, 38, 1, 2,
        36, 16, 1, 1, 27, 22, 36, 31, 32, 0
    ],
    "quiz_score": [
        59.7, 94.2, 82.2, 83.1, 72.2, 62.8, 43.3, 97.3, 78.1, 88.6,
        61.2, 86.6, 97.9, 59.2, 43.3, 72.4, 48.4, 74.1, 66.6, 54.6,
        67.3, 55.2, 68.7, 54.5, 67.1, 69.7, 47.2, 62.9, 81.4, 50.7,
        64.3, 53.8, 44.5, 88.3, 100.0, 67.1, 36.4, 59.6, 85.4, 80.9,
        68.6, 69.8, 35.0, 100.0, 60.6, 82.2, 50.8, 77.4, 55.7, 32.3,
        100.0, 77.8, 76.1, 74.3, 74.4, 100.0, 55.3, 56.6, 44.6, 47.0
    ]
})
```

---

## Dataset Description

| Column | Description |
|--------|--------------|
| study_hours | Student's study hours during the course |
| videos_watched | Number of educational videos watched |
| practice_solved | Number of solved practice exercises |
| quiz_score | Final quiz score (0–100) |

**Target Variable:**

- **quiz_score** (Regression)

---

## Part 1 – Implement

### a)

Split the dataset into:

- **80% Training**
- **20% Testing**

Use:

```python
random_state = 42
```

---

### b)

Train a **LinearRegression** model.

Use the following input features:

- study_hours
- videos_watched
- practice_solved

Target:

- quiz_score

---

### c)

Evaluate the model on the test set and calculate:

- MAE
- MSE
- R²

---

### d)

Plot an **Actual vs. Predicted** graph.

Also include the ideal prediction line:

```
y = x
```

---

### e)

Print the model coefficients (`coef_`) together with the corresponding feature names.

---

## Part 2 – Observe

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| MAE    |       |
| MSE    |       |
| R²     |       |

### Feature Coefficients

| Feature | Coefficient (coef_) |
|----------|-----------------------|
| study_hours     |  |
| videos_watched  |  |
| practice_solved |  |

---

## Part 3 – Reason

### a)

Based on the learned coefficients:

- Which feature has the greatest positive effect on the quiz score?
- Does this result match your intuition? Explain.

---

### b)

Examine the **Actual vs. Predicted** plot.

- Are most points close to the ideal line?
- If a few points are far from the line, what might this indicate?

---

### c)

A student claims:

> "My R² is 0.78, so my model is definitely reliable for predicting every new student."

Critique this statement.

Explain:

- What R² actually guarantees.
- What R² does **not** guarantee.

---

### d)

Suppose the **videos_watched** feature is removed from the model.

Would you expect the **R²** value to:

- Increase
- Decrease
- Stay approximately the same

Explain your reasoning based on the coefficient obtained for this feature.
