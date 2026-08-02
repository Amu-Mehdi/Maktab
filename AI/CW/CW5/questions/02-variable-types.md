# Problem 2: Identifying Variable Types

Identify the type of each of the following variables as **Continuous**, **Discrete**, or **Categorical**.

| Variable        | Continuous | Discrete | Categorical |
| --------------- | ---------- | -------- | ----------- |
| City            |            |          |             |
| DeliveryMinutes |            |          |             |
| ItemsCount      |            |          |             |
| PaymentMethod   |            |          |             |
| OrderValue      |            |          |             |
| Returned        |            |          |             |

Then answer the following questions:

1. Why is 'ItemsCount' considered a discrete variable?
   Answer: 'ItemsCount' represents the number of items ordered, which takes
   only integer values (e.g., 1, 2, 3, ...). You cannot order 2.5 items,
   so it has countable, distinct values with no intermediate values possible.
   ------------------------------

2. Why is 'OrderValue' considered a continuous variable?
   Answer: 'OrderValue' represents a monetary amount that can take any
   decimal value within a range. It is not limited to whole numbers and
   can have an infinite number of possible values between any two amounts.
   ------------------------------

3. Why does calculating the mean of the 'City' column not make sense?
   Answer: 'City' is a categorical (nominal) variable containing city names.
   These values are labels without numerical meaning or ordered relationships.
   Calculating the mean of city names is mathematically meaningless because
   you cannot add or average categorical labels like 'Tehran' and 'Isfahan'.
   ------------------------------

4. If the goal is to predict the order amount, which column should be the target?
   Answer: 'OrderValue' should be the target column because it represents
   the exact value we want to predict (the monetary amount of the order).
   ------------------------------

5. Suggest three columns that could be used as features for this problem.
   Answer: Three suitable features for predicting OrderValue are:
   - 'ItemsCount': More items typically mean higher order value
   - 'DeliveryMinutes': May correlate with order size/value
   - 'PaymentMethod': Different payment methods may indicate different order values
   Additional possible features: 'City', 'Returned'
   ------------------------------
