

# Question 5 – Bias-Variance Tradeoff Using Decision Trees

## Cheat Detection in an Online Video Game

Synthetic data is generated using **make_classification()**.

The goal is to classify whether a player is cheating.

---

## Dataset Generation

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=400,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=2,
    weights=[0.82, 0.18],   # class 0 = majority (normal players), class 1 = minority (cheaters)
    flip_y=0.05,            # small amount of label noise
    class_sep=0.9,
    random_state=42
)

# feature names for better readability
feature_names = ['accuracy_pct', 'reaction_time_ms', 'headshot_rate_pct', 'reports_count']
```

Features:

- accuracy_pct
- reaction_time_ms
- headshot_rate_pct
- reports_count

Target:

- is_cheater (1 = Cheater, 0 = Normal)

---

## Part 1 – Implement

### a)

Split the data:

- 70% Training
- 30% Testing

Use:

```python
random_state=42
```

---

### b)

Train three Decision Tree models:

```python
DecisionTreeClassifier(max_depth=1, random_state=42)   # Model 1
DecisionTreeClassifier(max_depth=4, random_state=42)   # Model 2
DecisionTreeClassifier(random_state=42)                 # Model 3
```

---

### c)

For each model, calculate:

- Training Accuracy
- Test Accuracy

---

## Part 2 – Observe

| Model | Train Accuracy | Test Accuracy | Train − Test Difference |
|-------|-----------------|-----------------|---------------------------|
| max_depth=1     |  |  |  |
| max_depth=4     |  |  |  |
| No depth limit  |  |  |  |

---

## Part 3 – Reason

### a)

Which model suffers from **Underfitting**?

Explain using the Training Accuracy.

---

### b)

Which model suffers from **Overfitting**?

Explain using the relationship between Train Accuracy and Test Accuracy.

---

### c)

Which model has the best **Generalization**?

Justify your answer using the Train-Test difference.

---

### d)

Relate the results to the **Bias-Variance Tradeoff**.

Explain:

- max_depth=1
- max_depth=None

---

### e)

Which model would you deploy?

Explain why Test Accuracy alone is insufficient.

---
