# K-Fold Cross Validation — From Fundamentals to Professional Machine Learning

> A complete, practical guide to Cross Validation: concepts, formulas, variants, Python implementation, Pipelines, Hyperparameter Tuning, Data Leakage, and Nested Cross Validation.

---

## 📑 Table of Contents

1. [Introduction](#1-introduction)
2. [K-Fold Cross Validation](#2-k-fold-cross-validation)
3. [How K-Fold Works](#3-how-k-fold-works)
4. [Mathematical Perspective](#4-mathematical-perspective)
5. [Manual Example](#5-manual-example)
6. [Choosing K](#6-choosing-k)
7. [Advantages and Limitations](#7-advantages-and-limitations)
8. [Hold-Out Validation](#8-hold-out-validation)
9. [Stratified K-Fold](#9-stratified-k-fold)
10. [Repeated K-Fold](#10-repeated-k-fold)
11. [Leave-One-Out Cross Validation](#11-leave-one-out-cross-validation)
12. [Group K-Fold](#12-group-k-fold)
13. [Time Series Cross Validation](#13-time-series-cross-validation)
14. [Nested Cross Validation](#14-nested-cross-validation)
15. [Comparison of Cross Validation Methods](#15-comparison-of-cross-validation-methods)
16. [K-Fold with NumPy](#16-k-fold-with-numpy)
17. [K-Fold with scikit-learn](#17-k-fold-with-scikit-learn)
18. [Stratified K-Fold in Python](#18-stratified-k-fold-in-python)
19. [Group K-Fold in Python](#19-group-k-fold-in-python)
20. [Time Series Cross Validation in Python](#20-time-series-cross-validation-in-python)
21. [Pipeline and Cross Validation](#21-pipeline-and-cross-validation)
22. [Hyperparameter Tuning](#22-hyperparameter-tuning)
23. [Nested Cross Validation in Python](#23-nested-cross-validation-in-python)
24. [Data Leakage](#24-data-leakage)
25. [Professional Mental Models](#25-professional-mental-models)
26. [Real-World Workflow](#26-real-world-workflow)
27. [Final Cheat Sheet](#27-final-cheat-sheet)
28. [Final Summary](#28-final-summary)

---

## 1. Introduction

Cross Validation is one of the most important tools for evaluating Machine Learning models.

But in real-world projects, the question isn't just:

> "How do I run K-Fold?"

We also need to know:

- Which type of Cross Validation fits the Dataset?
- When is plain K-Fold the wrong choice?
- When should we use Stratified K-Fold?
- What do we do if the data belongs to different entities (people, devices, etc.)?
- What do we do if the data is time-ordered?
- Where should Preprocessing be done?
- How do we combine Hyperparameter Tuning with Cross Validation?
- How does Data Leakage happen?
- When is Nested Cross Validation necessary?
- How do we fit Cross Validation into a real Machine Learning Pipeline?

This document focuses exactly on these questions — starting from a simple definition and building up to professional engineering decision-making.

---

## 2. K-Fold Cross Validation

### 2.1 Definition

In K-Fold Cross Validation, the Dataset is split into `K` parts (Folds).

At each step:

- One Fold is used for **Validation**.
- The remaining Folds are used for **Training**.

This process is repeated `K` times, so that every Fold plays the role of Validation exactly once.

For example, with 5-Fold:

```text
Fold 1 → Validation
Fold 2 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training
```

Then:

```text
Fold 2 → Validation
Fold 1 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training
```

And this pattern continues through Fold five.

---

## 3. How K-Fold Works

Suppose we have 10 samples:

```text
1 2 3 4 5 6 7 8 9 10
```

and:

```text
K = 5
```

The data is split like this:

```text
Fold 1 → 1  2
Fold 2 → 3  4
Fold 3 → 5  6
Fold 4 → 7  8
Fold 5 → 9 10
```

In the first iteration:

```text
Validation → Fold 1
Training   → Fold 2 + Fold 3 + Fold 4 + Fold 5
```

In the second iteration:

```text
Validation → Fold 2
Training   → Fold 1 + Fold 3 + Fold 4 + Fold 5
```

At the end, we have one Score per Fold:

```text
Score 1
Score 2
Score 3
Score 4
Score 5
```

and their average is usually computed as the final performance estimate.

---

## 4. Mathematical Perspective

Given the Scores from K Folds:

```text
S1, S2, ..., Sk
```

the Cross Validation mean is:

```text
CV Score = (S1 + S2 + ... + Sk) / K
```

### Example

```text
Fold 1 = 0.82
Fold 2 = 0.88
Fold 3 = 0.85
Fold 4 = 0.90
Fold 5 = 0.85
```

Average:

```text
(0.82 + 0.88 + 0.85 + 0.90 + 0.85) / 5
= 0.86
```

So:

```text
Mean CV Score = 0.86
```

### Standard Deviation

We can also check the Standard Deviation to see how much model performance varies across Folds.

```text
Mean = 0.86
Std  = 0.03
```

This means Fold performance was fairly close together — a stable estimate.

```text
Mean = 0.86
Std = 0.15
```

This can signal that the model is sensitive to how the data was split — worth investigating (e.g., outliers, imbalance, or a small dataset).

---

## 5. Manual Example

Suppose:

```text
Dataset = 100 samples
K = 5
```

In the balanced case:

```text
Fold Size = 100 / 5 = 20
```

At each iteration:

```text
Training   = 80 samples
Validation = 20 samples
```

### Key Point

Every sample:

- Is used in several Training Folds.
- Appears in Validation exactly once.

So every sample gets validated at least once — this is what makes K-Fold's estimate more reliable than a simple Hold-Out split.

---

## 6. Choosing K

Choosing K is a trade-off between:

- Amount of Training data
- Amount of Validation data
- Computational cost
- Bias
- Variance

Common values of K:

```text
K = 5
K = 10
```

These two are usually the most common, battle-tested choices in industry.

### Small K

For example:

```text
K = 2
```

The Training Set in each Fold is relatively small. This can make the estimate more sensitive to how the split was made (higher Bias).

### Large K

For example:

```text
K = 20
```

The model trains on more data in each Fold (lower Bias), but the number of training runs increases (higher cost).

### Very Large K

For example:

```text
K = N
```

This approaches LOOCV. In this case, the computational cost can become very high.

---

## 7. Advantages and Limitations

### ✅ Advantages

**Better use of the Dataset**
Compared to a single simple split, different samples get used across both Training and Validation.

**More stable estimate**
The result depends less on one particular split.

**Good for small and medium Datasets**
When the Dataset is limited, reusing the data multiple times is valuable.

**Good for Model Comparison**
Multiple models can be compared on the exact same splits.

### ⚠️ Limitations

**Computational cost**
If `K = 10`, the model is trained roughly 10 times. If training is expensive, the cost adds up fast.

**Data Leakage**
If preprocessing or feature selection is done incorrectly, Cross Validation can produce an unrealistic, overly optimistic result.

**Not suited to certain data structures**
Plain K-Fold may be a poor choice for:

- Grouped Data
- Time Series
- Some severely imbalanced Classification problems

---

## 8. Hold-Out Validation

### Definition

In Hold-Out, the Dataset is split once into a Training Set and a Validation Set. For example:

```text
80% Training
20% Validation
```

and the model is evaluated a single time.

### Intuition

Hold-Out is like giving one exam. It's fast, but the result can depend heavily on that one split.

### Advantages

- Fast
- Simple
- Low computational cost
- Good for very large Datasets

### Limitations

- Sensitive to the split
- The Validation set may not be representative of the whole Dataset
- Can be problematic on small Datasets

### When to Use

When the Dataset is very large and training the model is very expensive. For example:

```text
20,000,000 samples
```

with each training run taking several hours. In that situation, running 10-Fold CV can be prohibitively expensive.

### Example

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## 9. Stratified K-Fold

### Definition

Stratified K-Fold is a version of K-Fold that tries to preserve the class distribution across Folds.

Suppose:

```text
Class 0 = 90%
Class 1 = 10%
```

In Stratified K-Fold, each Fold roughly preserves this same ratio.

### Intuition

In a Classification problem with imbalanced classes, a random Fold might end up with very few examples of the minority class. Stratification tries to make each Fold a better representative of the overall class distribution.

### Example

Suppose:

```text
100 samples
90 → Class 0
10 → Class 1
```

With 5 Folds, we'd expect each Fold to have roughly:

```text
18 → Class 0
2  → Class 1
```

### Advantages

- Good for Classification
- Good for Class Imbalance
- Folds are better representatives of the class distribution

### Limitations

- Doesn't fully guarantee the imbalance problem is solved
- Not suited for Time Series
- Doesn't address Group dependency

### When to Use

For Classification, especially when classes are imbalanced.

### Python

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Then:

```python
for train_idx, val_idx in skf.split(X, y):
    ...
```

> Important: `skf.split(X, y)` needs `y` because it must inspect the class distribution.

---

## 10. Repeated K-Fold

### Definition

In Repeated K-Fold, K-Fold is run multiple times with different splits. For example:

```text
5-Fold × 3 Repeats
```

meaning three different sets of splits.

### Intuition

```text
Run 1 → Mean CV = 0.84
Run 2 → Mean CV = 0.87
Run 3 → Mean CV = 0.85
```

This shows how much model performance varies with the split.

### Advantages

- More stability
- Less dependence on a single random split
- Good for smaller Datasets

### Limitations

Training cost increases:

```text
5 Fold × 3 Repeat
= 15 Trainings
```

### Python

```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)
```

---

## 11. Leave-One-Out Cross Validation

### Definition

In LOOCV:

```text
K = N
```

meaning only one sample is used for Validation at each step. For example:

```text
Dataset = 100 samples
```

Number of training runs:

```text
100
```

### Example

```text
Iteration 1:
Train → 2 ... 100
Validation → 1

Iteration 2:
Train → 1,3,...,100
Validation → 2
```

and this continues to the end.

### Advantages

- Almost the entire Dataset is used for Training at each Fold
- Practical for very small Datasets

### Limitations

- Very expensive
- The number of training runs becomes very large
- The performance estimate can have high variance

### When to Use

When:

- The Dataset is very small
- Training the model is cheap
- Using nearly all data for Training matters

---

## 12. Group K-Fold

### Definition

Group K-Fold is used for data where samples belong to shared groups.

For example, in a medical dataset:

```text
Patient A:
Sample 1
Sample 2
Sample 3

Patient B:
Sample 4
Sample 5

Patient C:
Sample 6
Sample 7
Sample 8
```

Patient A's samples must not be scattered across both Train and Validation.

### Intuition

The model shouldn't see a given Patient during Training and then see the same Patient again during Validation. Otherwise, performance can look unrealistically good.

### Example

```python
groups = np.array([
    1, 1, 2, 2, 3,
    3, 4, 4, 5, 5
])
```

Then:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
```

and:

```python
for train_idx, val_idx in gkf.split(
    X,
    y,
    groups=groups
):
    ...
```

### Advantages

- Prevents Group Leakage
- Good for Patient-level Data
- Good for User-level Data
- Good for Device-level Data

### Limitations

- Folds may not have exactly equal sample counts
- If groups are very imbalanced, splitting becomes harder

### Examples of Groups

```text
Patient ID
Customer ID
User ID
Device ID
House ID
Subject ID
Company ID
```

---

## 13. Time Series Cross Validation

### Definition

For data that has a temporal order. For example:

```text
2020
2021
2022
2023
2024
2025
```

Future data must not leak into training that represents the past.

### Intuition

Validation should mimic real deployment conditions. If we're going to forecast the future, the model should be trained on the past and evaluated on the future.

### Example

```text
Fold 1

Train:
2020 2021

Validation:
2022
```

Next Fold:

```text
Train:
2020 2021 2022

Validation:
2023
```

And then:

```text
Train:
2020 2021 2022 2023

Validation:
2024
```

### Python

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5
)
```

Then:

```python
for train_idx, val_idx in tscv.split(X):
    ...
```

### Advantages

- Good for Forecasting
- Prevents Future Leakage
- Closer to real temporal deployment conditions

### Limitations

- The number of splits and the windowing strategy must fit the problem
- If Concept Drift is severe, CV alone isn't enough

---

## 14. Nested Cross Validation

### Definition

Nested CV has two layers of Cross Validation:

```text
Outer CV
    ↓
Evaluation

Inner CV
    ↓
Hyperparameter Tuning
```

### Why Nested CV?

Suppose we want to pick the best value of `C`:

```text
C = 0.01
C = 0.1
C = 1
C = 10
C = 100
```

If we use the same CV both to select the Hyperparameter and to report final performance, the performance can end up overly optimistic.

### Inner CV

Used to select:

- Hyperparameters
- Model
- Feature Selection
- Pipeline configuration

### Outer CV

Used to independently evaluate the selection process.

### Mental Model

> Inner selects; Outer judges.

### Example

```text
Outer Fold 1
    ↓
Inner CV
    ↓
C = 1
    ↓
Train
    ↓
Outer Validation
```

Then:

```text
Outer Fold 2
    ↓
Inner CV
    ↓
C = 10
    ↓
Train
    ↓
Outer Validation
```

Different Hyperparameters being chosen across different Outer Folds isn't necessarily a problem.

---

## 15. Comparison of Cross Validation Methods

| Method            | Speed      | Computational Cost | Bias                             | Variance                        | Suitable Data   | Main Use                 |
| ----------------- | ---------- | ------------------: | --------------------------------- | --------------------------------- | ---------------- | -------------------------- |
| Hold-Out          | Very fast  |                 Low | Relatively high                  | Relatively high                  | Very Large Data | Fast Evaluation          |
| K-Fold            | Moderate   |            Moderate | Low                               | Moderate                          | IID Data        | General CV               |
| Stratified K-Fold | Moderate   |            Moderate | Low                               | Moderate                          | Classification  | Class Balance             |
| Repeated K-Fold   | Slower     |                High | Low                               | Usually more stable               | IID Data        | Stability                 |
| LOOCV             | Very slow  |           Very high | Low                               | Can be high                       | Very Small Data | Small Dataset             |
| Group K-Fold      | Moderate   |            Moderate | Depends on data                   | Depends on data                   | Grouped Data     | Group Leakage Prevention |
| Time Series CV    | Moderate   |            Moderate | Depends on the window             | Depends on data drift             | Temporal Data    | Forecasting               |
| Nested CV         | Slow       |           Very high | Suited to evaluating selection    | More stable for the selection process | Model Selection  | Tuning + Evaluation      |

> **Note:** Bias and Variance in Cross Validation depend entirely on the Dataset, model, K, and data structure; the table above is a conceptual guide, not an absolute rule.

---

## 16. K-Fold with NumPy

Let's now implement K-Fold from scratch using NumPy.

### Dataset

```python
import numpy as np

X = np.array([
    [10, 1],
    [20, 2],
    [30, 3],
    [40, 4],
    [50, 5],
    [60, 6],
    [70, 7],
    [80, 8],
    [90, 9],
    [100, 10]
])

y = np.array([
    0, 0, 1, 0, 1,
    1, 0, 1, 1, 0
])
```

Here we have:

```text
10 Samples
2 Features
```

### Step 1 — Create Indices

```python
indices = np.arange(len(X))
```

Result:

```text
[0 1 2 3 4 5 6 7 8 9]
```

**Why indices?** Because we want to leave the original Dataset untouched and control the split via indices.

### Step 2 — Shuffle

```python
np.random.seed(42)

np.random.shuffle(indices)
```

For example:

```text
[8 1 5 0 7 2 9 4 3 6]
```

The seed ensures reproducibility.

### Step 3 — Number of Folds

```python
k = 5
```

### Step 4 — Split Indices

```python
folds = np.array_split(indices, k)
```

For example:

```text
Fold 1 → [8 1]
Fold 2 → [5 0]
Fold 3 → [7 2]
Fold 4 → [9 4]
Fold 5 → [3 6]
```

### Step 5 — Create Train and Validation Indices

```python
for i in range(k):

    val_idx = folds[i]

    train_idx = np.concatenate(
        [folds[j] for j in range(k) if j != i]
    )
```

At each iteration:

```text
Current Fold → Validation
Other Folds → Training
```

### Step 6 — Create Data

```python
X_train = X[train_idx]
X_val = X[val_idx]

y_train = y[train_idx]
y_val = y[val_idx]
```

Important: `X[i]` and `y[i]` must always belong to the same sample.

### Complete NumPy Implementation

```python
import numpy as np

X = np.array([
    [10, 1],
    [20, 2],
    [30, 3],
    [40, 4],
    [50, 5],
    [60, 6],
    [70, 7],
    [80, 8],
    [90, 9],
    [100, 10]
])

y = np.array([
    0, 0, 1, 0, 1,
    1, 0, 1, 1, 0
])

k = 5

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

folds = np.array_split(indices, k)

for i in range(k):

    val_idx = folds[i]

    train_idx = np.concatenate(
        [folds[j] for j in range(k) if j != i]
    )

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"Fold {i + 1}")
    print("Train indices:", train_idx)
    print("Validation indices:", val_idx)
    print()
```

### Adding a Machine Learning Model

For example, Logistic Regression:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

scores = []

for i in range(k):

    val_idx = folds[i]

    train_idx = np.concatenate(
        [folds[j] for j in range(k) if j != i]
    )

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    score = accuracy_score(
        y_val,
        predictions
    )

    scores.append(score)
```

Finally:

```python
print(scores)
print(np.mean(scores))
```

### Why the Model Must Be Recreated Each Time

Correct approach:

```python
for ...:

    model = LogisticRegression()

    model.fit(...)
```

**Not** this:

```python
model = LogisticRegression()

for ...:
    model.fit(...)
```

**Reason:** Each Fold must have an independent training run. We don't want the model's state to carry over from one Fold to the next.

---

## 17. K-Fold with scikit-learn

In real projects, it's better to use the standard tooling.

```python
from sklearn.model_selection import KFold

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### Getting the Folds

```python
for train_idx, val_idx in kf.split(X):

    print("Train:", train_idx)
    print("Validation:", val_idx)
```

`scikit-learn` handles the fold-building logic for us.

### cross_val_score

Instead of writing the loop yourself:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="accuracy"
)

print("Scores:", scores)
print("Mean:", scores.mean())
```

Under the hood, this happens:

```text
Split
↓
Train
↓
Predict
↓
Metric
↓
Repeat
↓
Aggregate
```

---

## 18. Stratified K-Fold in Python

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Usage:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)
```

---

## 19. Group K-Fold in Python

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(
    n_splits=5
)
```

Usage:

```python
for train_idx, val_idx in gkf.split(
    X,
    y,
    groups=groups
):
    ...
```

### Checking Group Leakage

```python
for train_idx, val_idx in gkf.split(
    X,
    y,
    groups=groups
):

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    overlap = train_groups.intersection(
        val_groups
    )

    print("Overlap:", overlap)
```

Correct result:

```text
set()
```

meaning there's no shared group between Train and Validation.

### StratifiedGroupKFold

When we have both **Group dependency** and **Class imbalance** at the same time:

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

and:

```python
for train_idx, val_idx in sgkf.split(
    X,
    y,
    groups=groups
):
    ...
```

This method tries to:

1. Keep groups separated.
2. Keep the class distribution as balanced as possible.

---

## 20. Time Series Cross Validation in Python

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5
)
```

Usage:

```python
for train_idx, val_idx in tscv.split(X):

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
```

### Checking the Indices

```python
for fold, (train_idx, val_idx) in enumerate(
    tscv.split(X)
):

    print(f"Fold {fold + 1}")
    print("Train:", train_idx)
    print("Validation:", val_idx)
```

The goal is to preserve the temporal order.

---

## 21. Pipeline and Cross Validation

This section is one of the **most important** parts of the whole topic.

Suppose we have `StandardScaler` + `LogisticRegression`.

### ❌ The Wrong Way

```python
scaler.fit(X)

X_scaled = scaler.transform(X)

scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5
)
```

**The problem:** `StandardScaler` has seen the entire Dataset. As a result, Validation information has leaked into the preprocessing step. This is a form of **Data Leakage**.

### ✅ The Correct Way — Using a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

Then:

```python
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
```

### What Happens Inside the Pipeline

For each Fold:

```text
Training Data
      ↓
Fit Scaler
      ↓
Transform Training
      ↓
Fit Model
      ↓
Validation Data
      ↓
Transform using Training Scaler
      ↓
Predict
      ↓
Calculate Metric
```

Validation data is never involved in fitting the Scaler.

### Why Pipelines Matter Beyond Scaling

A Pipeline isn't just for scaling. We can have:

```text
Imputation
↓
Scaling
↓
PCA
↓
Feature Selection
↓
Model
```

For example:

```python
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

Any step that learns from the data must sit on the correct side of the Cross Validation boundary.

---

## 22. Hyperparameter Tuning

Suppose we have Logistic Regression and want to tune the `C` parameter.

```python
param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}
```

### Why `model__C`?

Because we have a Pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

```text
Step name      → model
Parameter name → C
So             → model__C
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)
```

Then:

```python
grid.fit(X, y)
```

Best parameter:

```python
print(grid.best_params_)
```

Best CV score:

```python
print(grid.best_score_)
```

### ⚠️ An Important Note About `best_score_`

Suppose we tested `C = 0.01, 0.1, 1, 10, 100` and CV picked the best value.

If we report that same score as a fully independent estimate of final performance, we risk **Optimistic Bias** — because the model/hyperparameter was selected using that same data.

The more professional approach: **Nested Cross Validation**.

---

## 23. Nested Cross Validation in Python

```python
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)
```

**Outer:**

```python
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

**Inner:**

```python
inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)
```

**Grid Search:**

```python
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="accuracy"
)
```

**Outer Evaluation:**

```python
nested_scores = cross_val_score(
    grid,
    X,
    y,
    cv=outer_cv,
    scoring="accuracy"
)
```

### Nested CV Flow

```text
Outer Fold 1
│
├── Outer Training
│
│   └── Inner CV
│       ├── Try C=0.01
│       ├── Try C=0.1
│       ├── Try C=1
│       ├── Try C=10
│       └── Try C=100
│
│   → Best C
│
└── Evaluate on Outer Validation
```

Then Outer Fold 2 repeats the same process. We might see:

```text
Outer Fold 1 → C = 1
Outer Fold 2 → C = 10
Outer Fold 3 → C = 1
Outer Fold 4 → C = 0.1
Outer Fold 5 → C = 1
```

This isn't necessarily a problem — it just shows that the chosen Hyperparameter can be sensitive to the data.

### Final Model After Nested CV

Nested CV is mainly for estimating the performance of the model-selection *process*, not for producing the final model. Afterward:

```text
Nested CV
      ↓
Understand expected performance
      ↓
Select final configuration
      ↓
Train final model on Development Data
      ↓
Evaluate on Independent Test
```

We shouldn't just take one of the Outer-Fold models and ship it to production as-is.

---

## 24. Data Leakage

Data Leakage means information that shouldn't be available during Training somehow makes its way into the Training or Model Selection process. This is one of the **most dangerous** problems in Machine Learning.

### Scaling Leakage

**Wrong:**

```python
scaler.fit(X)

X_scaled = scaler.transform(X)

cross_val_score(
    model,
    X_scaled,
    y,
    cv=5
)
```

**The problem:** the Scaler has seen the whole Dataset.

**The correct way:**

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", model)
])
```

### Imputation Leakage

Suppose we have missing values. Dangerous approach:

```python
imputer.fit(X)
X_imputed = imputer.transform(X)
```

...followed by CV. The imputer has seen the entire Dataset's information.

**The correct way:**

```python
Pipeline([
    ("imputer", SimpleImputer()),
    ("model", model)
])
```

### PCA Leakage

**Wrong approach:**

```text
Entire Dataset
↓
Fit PCA
↓
Cross Validation
```

**Correct approach:**

```text
Fold Training Data
↓
Fit PCA
↓
Transform Training
↓
Transform Validation
```

A Pipeline manages this process for you.

### Feature Selection Leakage

**Wrong approach:**

```text
Entire Dataset
↓
Feature Selection
↓
Cross Validation
```

**Correct approach:**

```text
Training Fold
↓
Feature Selection
↓
Model
↓
Validation Fold
```

### Group Leakage

Suppose:

```text
Patient A
├── Sample 1
├── Sample 2
└── Sample 3
```

If:

```text
Train → Sample 1, Sample 2
Validation → Sample 3
```

the model has already seen this same Patient before.

**The correct approach:** set `Group = Patient ID` and use `GroupKFold`.

### Time Leakage

If:

```text
2025 → Training
2024 → Validation
```

the model has received future information to predict the past.

**Correct approach:**

```text
Past → Training
Future → Validation
```

### Hyperparameter Leakage

Suppose:

```text
CV
↓
Select Best Hyperparameter
↓
Same CV Score
↓
Report as Final Performance
```

This evaluation isn't truly independent.

**Professional approach:**

```text
Inner CV → Selection
Outer CV → Evaluation
```

---

## 25. Professional Mental Models

### K-Fold

**Mental Model:** Give the model several different exams.
**Analogy:** Don't judge a student on a single exam.

```text
Dataset
 ↓
Exam 1
 ↓
Exam 2
 ↓
Exam 3
 ↓
Exam 4
 ↓
Exam 5
```

> K-Fold reduces the dependency of the result on any single split.

### Stratified K-Fold

**Mental Model:** Every Fold should represent the classes fairly.
**Analogy:** If a classroom is 90% Student A and 10% Student B, every quiz group should roughly reflect that same mix.

> Stratification preserves class proportions.

### Group K-Fold

**Mental Model:** Don't split an entity in half.
**Analogy:** If you've seen a Patient during Training, that same Patient shouldn't show up in Validation.

> Groups must stay separated between Train and Validation.

### Time Series CV

**Mental Model:** Don't show the future to the past.
**Analogy:** If you want to predict tomorrow, you shouldn't use tomorrow's information to train today's model.

> Validation should represent the future the model will actually face in the real world.

### Nested CV

**Mental Model:** One selects, one judges.
**Analogy:** A judge doesn't pick the winner and then evaluate their own decision — we keep Selection and Evaluation separate.

> Inner CV selects; Outer CV evaluates the selection process itself.

### Pipeline

**Mental Model:** Anything that "learns" should only learn from Training data.
**Analogy:** If you want to give a real exam, you shouldn't show the student the answer sheet beforehand.

> Every step that gets `fit` must sit on the correct side of the Training boundary.

---

## 26. Real-World Workflow

### Example 1 — House Price Prediction (Regression)

**Features:** `Area, Bedrooms, Bathrooms, Age, Location, Garage`
**Target:** `Price`
**Data:** `100,000 houses`

Since this is Regression and there's no Group or Time structure, `KFold` is a good fit.

**Pipeline:**

```text
Imputation → Scaling → Regression Model
```

```python
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error"
)

param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=kf,
    scoring="neg_mean_absolute_error"
)

grid.fit(X, y)
```

If we set aside an independent Test set from the start:

```text
Development Data
↓
CV + Tuning
↓
Final Model
↓
Independent Test
```

The Test Set should never be used to select Hyperparameters.

### Example 2 — Disease Detection (Grouped Data)

**Dataset:** `Patient ID, Age, Blood Pressure, Lab Results, Symptoms, Target`

Each Patient may have multiple Visits/Records. With plain KFold, `Visit 1 → Train` and `Visit 2 → Validation` could come from the same patient — that's leakage.

**Correct approach:**

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
```

And if we also have Class Imbalance (e.g., `Healthy = 95%`, `Disease = 5%`):

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### Example 3 — Sales Forecasting (Time Series)

**Dataset:** `Date, Advertising, Price, Promotion, Sales`

Since time matters here, plain KFold can be problematic (e.g., `2025 → Training`, `2024 → Validation`).

**Correct approach:**

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

### K-Fold vs. Train/Test Split

```text
Train/Test Split                 K-Fold
─────────────────                 ──────────────────
Train  ████████████████           Fold 1 → Validation
Test   ████                       Fold 2 → Validation
                                   Fold 3 → Validation
Split happens once.                Fold 4 → Validation
                                   Fold 5 → Validation
                                   Every sample is validated once.
```

**Train/Test Split is better when:** the Dataset is very large and training is very expensive.

**K-Fold is better when:** the Dataset is small/medium, you want a more stable estimate, model training is affordable, and you want to compare models more reliably.

### Professional Decision Framework

Before choosing a Cross Validation strategy, ask these questions:

| # | Question | Answer |
|---|------|------|
| 1 | Is there temporal dependency in the data? | ✅ Time Series CV |
| 2 | Do multiple samples belong to the same entity? | ✅ Group K-Fold |
| 3 | Is it Classification with Class Imbalance? | ✅ Stratified K-Fold |
| 4 | Both Groups and Class Imbalance? | ✅ Stratified Group K-Fold |
| 5 | Need Hyperparameter Tuning + independent evaluation? | ✅ Nested CV |
| 6 | Is the Dataset huge and training very expensive? | ✅ Hold-Out |

### Important Engineering Rule

Always ask before running CV:

> **What is the structure of the data?**

Not simply:

> "I always use KFold with `k=5`."

---

## 27. Final Cheat Sheet

### General

```text
IID Data                          → KFold
Classification                    → StratifiedKFold
Groups                            → GroupKFold
Groups + Class Imbalance          → StratifiedGroupKFold
Time                              → TimeSeriesSplit
Repeated Stability Check          → RepeatedKFold
Very Small Dataset                → LOOCV
Huge Dataset + Expensive Model    → Hold-Out
Hyperparameter Selection
+ Independent Evaluation          → Nested CV
```

### Pipeline

Anything that gets `fit` — `Imputer, Scaler, PCA, Feature Selection, Encoder, Model` — must sit on the correct side of the CV boundary:

```python
Pipeline([
    (...),
    (...),
    ("model", model)
])
```

### Leakage

```text
❌ Never:
Fit preprocessing on entire Dataset
↓
Cross Validation

✅ Do:
Cross Validation
↓
Fit preprocessing on Training Fold
↓
Transform Validation Fold
```

### Nested CV

```text
Outer CV → Independent Evaluation
Inner CV → Hyperparameter Tuning
```

> Inner selects, Outer evaluates.

### Time Series

```text
✅ Past → Train
✅ Future → Validation

❌ Never: Future → Train, Past → Validation
```

### Groups

```text
✅ Patient A → Train
✅ Patient B → Train
✅ Patient C → Validation

❌ Patient A Sample 1 → Train
❌ Patient A Sample 2 → Validation
```

### Code Reference

```python
# KFold
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Group
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)

# Time Series
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)

# Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=cv)

# Pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Grid Search
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=5)
```

### Final Professional Pipeline

```text
                         RAW DATA
                            │
                            ▼
                    Understand Dataset
                            │
                            ▼
                  Identify Data Structure
                            │
              ┌─────────────┼─────────────┐
              │             │             │
             IID          Groups         Time
              │             │             │
           K-Fold       Group K-Fold   Time Series
              │
              ▼
       Classification?
          /         \
        Yes          No
         │            │
    Stratified      K-Fold
         │
         ▼
      Pipeline
         │
         ├── Imputation
         │
         ├── Scaling
         │
         ├── Feature Engineering
         │
         ├── Feature Selection
         │
         └── Model
              │
              ▼
      Hyperparameter Tuning
              │
              ▼
          Nested CV?
          /        \
        Yes         No
         │           │
     Inner CV     Regular CV
         │           │
         └─────┬─────┘
               ▼
        Model Selection
               │
               ▼
        Final Training
               │
               ▼
       Independent Test
               │
               ▼
          Deployment
```

---

## 28. Final Summary

K-Fold Cross Validation isn't just a technique for splitting a Dataset. In professional Machine Learning, Cross Validation is part of a much larger system:

```text
Data
↓
Data Structure
↓
Validation Strategy
↓
Preprocessing
↓
Pipeline
↓
Model
↓
Hyperparameter Tuning
↓
Cross Validation
↓
Model Selection
↓
Final Training
↓
Independent Test
↓
Deployment
```

The most important point is that **Cross Validation must match the structure of the data**:

| Data Structure | Right Method |
|---|---|
| IID | K-Fold |
| Classification + Class Imbalance | Stratified K-Fold |
| Grouped Samples | Group K-Fold |
| Groups + Imbalance | Stratified Group K-Fold |
| Time-dependent | Time Series CV |
| Need a stability check | Repeated K-Fold |
| Very small Dataset | LOOCV |
| Very large Dataset + expensive training | Hold-Out |
| Hyperparameter Tuning + reliable evaluation | Nested Cross Validation |

The most important risk throughout this whole process is **Data Leakage** — especially around:

```text
Scaling
Imputation
PCA
Feature Selection
Feature Engineering
Encoding
Hyperparameter Tuning
Groups
Time
```

### The Golden Rule

> **Anything that learns from the data must only learn from the Training data.**

This is exactly why `Pipeline` is one of the most important tools for using Cross Validation professionally.

Ultimately, a professional Machine Learning Engineer doesn't just ask:

> "What should K be?"

They ask:

> "What is the structure of the data? What type of validation fits it? What could cause leakage? Where should preprocessing be fit? How should hyperparameters be selected? And how should final performance be evaluated independently?"

That's the difference between **knowing K-Fold** and **using Cross Validation professionally in a real Machine Learning project**. 🚀
