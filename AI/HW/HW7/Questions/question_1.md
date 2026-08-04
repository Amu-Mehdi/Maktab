## Question 1 – Problem Type Identification, Model Selection, and Evaluation Metric

### Part 1 – Implement / Observe

Below are six scenarios from different domains.

For each scenario:

1. Determine whether the problem is **Regression**, **Classification**, or **Clustering**.
2. Select the most appropriate algorithm from the following:
   - Linear Regression
   - Logistic Regression
   - KNN Regressor
   - KNN Classifier
   - Decision Tree Classifier
   - K-Means
3. Select the most appropriate evaluation metric from the following:
   - R²
   - MAE
   - MSE
   - Precision / Recall / F1-score
   - Accuracy
   - Confusion Matrix

> **Note:** If classification evaluation metrics are not applicable, leave the evaluation metric blank.

### Scenarios

1. Predict the amount of time (in minutes) that a user watches an online educational video before exiting.

2. Determine whether a comment posted on an online store product is **Positive** or **Negative**.

3. Predict the daily energy consumption of an office building based on temperature, number of employees, and building area.

4. Determine whether a bank card transaction is fraudulent or legitimate.

5. Predict the estimated arrival time of a postal shipment based on distance and shipping method.

6. Categorize customers of an online store into behavioral groups without having any pre-existing labels.

### Complete the Following Table

| Scenario | Problem Type | Suggested Algorithm | Evaluation Metric |
|----------|--------------|----------------------|--------------------|
| 1        |              |                      |                    |
| 2        |              |                      |                    |
| 3        |              |                      |                    |
| 4        |              |                      |                    |
| 5        |              |                      |                    |
| 6        |              |                      |                    |

---

### Part 2 – Reasoning

#### a)

Compare **Scenario 6** (customer categorization without labels) with **Scenario 2** (positive/negative comment detection).

Why can't evaluation metrics such as **Accuracy** or **F1-score** be used for Scenario 6?

---

#### b)

In **Scenario 4** (fraud detection), fraudulent transactions are much less frequent than legitimate ones.

Why is a high **Accuracy** alone not sufficient to indicate that the model performs well?

> A conceptual explanation is sufficient.

---

#### c)

Explain the main difference between the output of a **Regression** model and a **Classification** model using **Scenarios 1 and 2** as examples.

How does this difference affect the choice of evaluation metrics?

---
