

# Question 3 – Logistic Regression and Evaluation on Imbalanced Data

## Predicting Membership Churn in a Sports Club

A sports club wants to identify subscribers who are likely to cancel their membership before they actually churn.

The dataset contains **90 subscribers** and is intentionally **imbalanced**, similar to many real-world business datasets.

```python
import pandas as pd

gym_data = pd.DataFrame({
    "visits_per_month": [
        6, 5, 0, 4, 15, 12, 18, 4, 17, 6,
        6, 0, 11, 18, 8, 0, 16, 12, 16, 14,
        17, 13, 3, 19, 13, 15, 15, 4, 0, 16,
        18, 15, 0, 11, 4, 0, 0, 5, 17, 15,
        17, 10, 0, 6, 5, 10, 14, 4, 13, 19,
        6, 7, 17, 4, 5, 17, 3, 7, 0, 12,
        19, 12, 1, 14, 13, 9, 11, 6, 8, 19,
        12, 8, 10, 17, 14, 5, 12, 7, 2, 4,
        14, 12, 9, 16, 11, 9, 3, 10, 4, 16
    ],
    "avg_session_min": [
        51.1, 40.8, 15.4, 74.2, 42.2, 44.3, 26.8, 54.0, 66.4, 56.7,
        25.1, 18.6, 66.3, 73.9, 48.1, 13.9, 40.4, 27.1, 49.5, 51.0,
        42.2, 51.2, 13.4, 43.6, 54.7, 49.9, 74.3, 47.5, 11.3, 25.4,
        51.7, 47.2, 12.6, 32.4, 54.5, 11.0, 13.1, 28.6, 33.4, 45.6,
        38.9, 64.9, 24.3, 58.1, 42.0, 72.3, 66.1, 53.3, 64.1, 25.1,
        68.6, 52.4, 49.4, 63.6, 63.1, 62.6, 17.9, 71.0, 13.2, 50.1,
        59.8, 48.8, 22.6, 27.8, 49.8, 30.3, 51.6, 56.8, 33.3, 54.3,
        31.1, 52.9, 35.4, 65.0, 37.1, 47.1, 73.8, 59.7, 12.0, 59.5,
        72.2, 52.2, 62.6, 51.2, 37.1, 46.4, 16.8, 39.4, 37.3, 42.7
    ],
    "days_since_last_visit": [
        8, 10, 21, 1, 10, 11, 8, 1, 14, 3,
        7, 41, 1, 6, 14, 20, 8, 7, 12, 14,
        2, 14, 23, 12, 7, 4, 10, 5, 46, 4,
        11, 4, 58, 11, 10, 26, 49, 4, 14, 5,
        13, 5, 43, 13, 10, 7, 1, 5, 2, 13,
        14, 11, 1, 12, 3, 7, 27, 3, 25, 9,
        1, 5, 27, 11, 8, 4, 9, 0, 9, 5,
        12, 10, 5, 1, 4, 2, 12, 10, 55, 10,
        8, 14, 13, 8, 3, 11, 34, 12, 6, 8
    ],
    "has_personal_trainer": [
        1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
        1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 0, 0, 0, 0
    ],
    "churned": [
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0
    ]
})
```

---

## Dataset Description

| Column | Description |
|--------|--------------|
| visits_per_month | Number of club visits during one month |
| avg_session_min | Average duration of each visit (minutes) |
| days_since_last_visit | Number of days since the last visit |
| has_personal_trainer | Personal trainer status (1 = Yes, 0 = No) |
| churned | Target variable (1 = Cancelled, 0 = Active) |

---

## Part 1 – Implement

### a)

- Display the first **five rows** of the dataset.
- Count the number of samples in each class (`0` and `1`) of the **churned** column.
- Report the percentage of each class.

---

### b)

Split the dataset into:

- 80% Training
- 20% Testing

Use:

- `random_state=42`
- `stratify=y`

to preserve class proportions.

---

### c)

- Standardize the features using **StandardScaler**.
- Train a **LogisticRegression** model.

---

### d)

Evaluate the model on the test set by computing:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score

---

## Part 2 – Observe

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| Accuracy  |  |
| Precision |  |
| Recall    |  |
| F1-score  |  |

### Confusion Matrix

| TP | TN | FP | FN |
|----|----|----|----|
|    |    |    |    |

---

## Part 3 – Reason

### a)

Based on the class percentages calculated earlier:

- Is the dataset balanced?
- If a dummy model always predicts **"Not Churned (0)"**, what would its approximate Accuracy be?

Explain without writing code.

---

### b)

Compare your trained model with the dummy model.

Explain:

- Why Accuracy alone is misleading.
- Whether Precision or Recall is more important.
- Why missing an actual churned customer (FN) is usually more costly than incorrectly warning a loyal customer (FP).

---

### c)

Suppose the club sends a discount code to every customer predicted to churn.

Which error is less costly?

- False Positive (FP)
- False Negative (FN)

Explain.

---

### d)

Obtain prediction probabilities on the test set.

Reduce the decision threshold from **0.50** to **0.30**.

Explain what you expect to happen to:

- Precision
- Recall

Then verify your answer by running the model.

---