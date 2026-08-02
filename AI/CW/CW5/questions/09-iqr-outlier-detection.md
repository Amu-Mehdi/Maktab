# Problem 9: Outlier Detection Using the IQR Method

For the `OrderValue` column, perform the following steps:

1. Calculate Q1.
2. Calculate Q3.
3. Calculate the IQR using the difference between Q3 and Q1.
4. Calculate the lower bound.
5. Calculate the upper bound.
6. Use a Pandas filter to extract the orders whose values fall outside these two bounds.
7. Display only the first five rows of the detected outliers.

Based on the output, identify the outliers in the `OrderValue` column using the IQR method.

Then answer the following questions:

1. How many outliers were identified, and what is the corresponding `OrderID`?
2. What is the value of the outlier order, and is it above the upper bound or below the lower bound?
3. Why does this value have a greater effect on the mean than on the median?
4. Does being identified as an outlier necessarily mean that the value is incorrect? Give a brief reason.
