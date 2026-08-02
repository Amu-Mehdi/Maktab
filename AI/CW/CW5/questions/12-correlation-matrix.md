# Problem 12: Correlation Matrix

Select only the numerical columns of the dataset using:

```python
select_dtypes(include="number")
```

Then:

1. Calculate the correlation matrix.
2. Display the correlation matrix using `imshow()`.

Based on the output, answer the following questions:

1. Write the names of the columns included in the correlation matrix.
2. Why are columns such as `City` and `PaymentMethod` not included in the matrix?
3. Why are all values on the main diagonal of the matrix equal to `1`?
4. Which variable has the strongest positive correlation with `OrderValue`?
5. Which variable has the strongest negative correlation with `OrderValue`?
6. Why should the correlation of `OrderValue` with itself be excluded from the list of possible features?
7. Does the variable with the strongest correlation necessarily cause an increase in `OrderValue`?
8. Should the final decision about feature selection be based only on the Correlation Matrix?
