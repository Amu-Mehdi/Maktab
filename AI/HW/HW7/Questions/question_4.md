

# Question 4 – KNN Regression for Flight Ticket Price Prediction

## Investigating the Effect of K and Feature Standardization

An airline reservation system wants to predict ticket prices using K-Nearest Neighbors Regression.

The dataset contains **55 flights**.

```python
import pandas as pd

flight_data = pd.DataFrame({
    "days_before_flight": [
        26, 64, 81, 82, 56, 14, 77, 34, 72, 83,
        25, 49, 33, 46, 5, 35, 13, 2, 75, 38,
        85, 82, 25, 18, 68, 71, 44, 65, 80, 24,
        28, 71, 11, 58, 28, 45, 15, 8, 56, 80,
        65, 41, 28, 6, 54, 72, 51, 60, 2, 4,
        9, 88, 82, 69, 73
    ],
    "distance_km": [
        575, 2245, 1234, 3510, 2458, 1752, 564, 1229, 685, 1291,
        4245, 718, 319, 3456, 2142, 743, 793, 1140, 1362, 2259,
        419, 4258, 4171, 3104, 3216, 3504, 2533, 2669, 2088, 4486,
        930, 4032, 662, 3818, 4016, 3190, 4378, 1985, 1231, 2813,
        2866, 4244, 2077, 614, 3949, 1424, 4340, 1105, 1929, 2600,
        3550, 4009, 2305, 2051, 2358
    ],
    "num_stops": [
        0, 1, 0, 1, 0, 0, 2, 2, 2, 0,
        0, 2, 0, 1, 1, 1, 0, 1, 2, 2,
        2, 2, 0, 1, 2, 2, 0, 0, 2, 1,
        1, 2, 0, 1, 1, 2, 2, 1, 2, 2,
        1, 1, 0, 2, 1, 2, 1, 0, 1, 0,
        2, 0, 2, 2, 1
    ],
    "ticket_price_usd": [
        335, 380, 275, 391, 458, 431, 61, 331, 133, 233,
        683, 214, 308, 506, 476, 298, 402, 480, 162, 390,
        63, 425, 705, 569, 368, 400, 474, 375, 233, 703,
        354, 443, 380, 492, 571, 475, 663, 495, 207, 287,
        401, 595, 453, 357, 544, 189, 552, 254, 476, 609,
        611, 456, 224, 246, 315
    ]
})
```

---

## Dataset Description

| Column | Description |
|--------|--------------|
| days_before_flight | Days remaining until departure |
| distance_km | Flight distance (km) |
| num_stops | Number of intermediate stops |
| ticket_price_usd | Ticket price (Regression Target) |

---

## Part 1 – Implement

### a)

Split the data:

- 80% Training
- 20% Testing

Use:

```python
random_state=42
```

---

### b)

Standardize all input features using **StandardScaler**.

Train a:

```python
KNeighborsRegressor(n_neighbors=5)
```

---

### c)

Evaluate the model using:

- MAE
- MSE
- R²

---

### d)

Retrain the model using:

- K = 1
- K = 3
- K = 5
- K = 7
- K = 9
- K = 15

Record the R² value for each.

---

### e)

Predict the ticket price for:

| Feature | Value |
|---------|-------|
| days_before_flight | 20 |
| distance_km | 2000 |
| num_stops | 1 |

Remember to standardize this sample using the training scaler.

---

## Part 2 – Observe

### Evaluation Metrics

| MAE | MSE | R² |
|-----|-----|-----|
|     |     |     |

### Effect of K

| K | R² |
|---|-----|
| 1  |  |
| 3  |  |
| 5  |  |
| 7  |  |
| 9  |  |
| 15 |  |

---

## Part 3 – Reason

### a)

From the K vs R² table:

- Which K performs best?
- Is the relationship monotonic, or does it have an optimal point?

---

### b)

Explain:

- Why K=1 is highly sensitive to noise.
- What happens if K equals the total number of training samples.

---

### c)

Why is feature standardization essential in KNN?

Explain why Euclidean distance becomes dominated by **distance_km** if the data is not standardized.

---

### d)

Why is KNN called a **Lazy Learning** algorithm?

How does this affect prediction time when the dataset becomes very large?

---