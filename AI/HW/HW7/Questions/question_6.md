

# Question 6 – K-Means Clustering of Music Listener Behavior

## Discovering Hidden Patterns Without Labels

A music streaming platform wants to group users according to listening behavior.

The dataset contains **48 users**.

```python
import pandas as pd

listener_data = pd.DataFrame({
    "weekly_listening_hours": [
        9.4, 3.0, 4.6, 29.3, 27.2, 9.6, 31.9, 16.8, 8.2, 12.3,
        4.7, 8.8, 32.3, 8.2, 21.8, 1.4, 5.8, 25.3, 22.7, 22.2,
        15.7, 2.2, 31.5, 25.1, 2.5, 12.3, 32.4, 4.7, 4.0, 25.9,
        4.8, 3.3, 2.2, 4.4, 11.1, 12.7, 17.6, 11.5, 24.2, 17.1,
        23.9, 14.7, 5.0, 3.1, 17.7, 20.3, 30.2, 5.5
    ],
    "followed_genres_count": [
        5, 1, 3, 11, 14, 7, 11, 7, 4, 8,
        1, 4, 12, 7, 12, 3, 1, 14, 11, 10,
        6, 2, 12, 12, 1, 6, 14, 1, 3, 8,
        3, 3, 1, 2, 6, 5, 4, 5, 12, 5,
        14, 5, 2, 3, 7, 8, 12, 3
    ]
})
```

---

## Dataset Description

| Column | Description |
|--------|--------------|
| weekly_listening_hours | Hours listened per week |
| followed_genres_count | Number of followed music genres |

> There is **no target column**.

---

## Part 1 – Implement

### a)

Display:

- First five rows
- Total number of samples

---

### b)

Standardize the features.

---

### c)

Train:

```python
KMeans(n_clusters=3, random_state=42)
```

---

### d)

Add cluster labels to the dataset.

Create a Scatter Plot showing:

- Colored clusters
- Cluster centroids

---

### e)

Print the characteristics of each cluster center.

For better interpretation, also report the average original feature values of each cluster.

---

## Part 2 – Observe

| Cluster | Members | Avg Weekly Listening Hours | Avg Followed Genres |
|---------|---------|------------------------------|-------------------------|
| 0 |  |  |  |
| 1 |  |  |  |
| 2 |  |  |  |

---

## Part 3 – Reason

### a)

Assign meaningful names to each cluster based on the observed averages.

Example:

- Occasional Listener
- Moderate Listener
- Heavy Multi-Genre Listener

---

### b)

Do cluster numbers (0, 1, 2) indicate ranking or quality?

Explain why labels may change between runs.

---

### c)

Why can't Accuracy or F1-score evaluate clustering quality?

Compare this with Question 3.

---

### d)

Run KMeans with:

- n_clusters=2
- n_clusters=4

Discuss:

- Does more clusters always mean better clustering?
- What happens if n_clusters equals the number of samples (48)?