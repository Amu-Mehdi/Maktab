# Problem 3: Understanding the Problem and Selecting Features

Read the dataset using Pandas and display the first five rows and the general information about the dataset.

## Problem 1: Predicting `FinalScore`

1. Identify the Target column.
2. Select the numerical columns of the dataset.
3. Calculate the correlation of the numerical columns with `FinalScore`.
4. Examine the columns based on the strength of their relationship with `FinalScore`.
5. Draw a Scatter Plot for the three columns that appear to have the strongest relationships with `FinalScore`.
6. Select three appropriate Features based on:

   * Their correlation with `FinalScore`
   * The shape of the relationship in the Scatter Plot
   * Whether the relationship between the column and the final score makes logical sense
7. Provide a short reason for each selected Feature.
8. Determine whether this is a Regression or Classification problem and explain your reasoning.

## Feature Selection Notes

When selecting Features, consider the following:

* `StudentID` is only a student identifier and usually does not provide useful information for prediction.
* `FinalScore` should not be selected as a Feature because it is the Target of the problem.
* `Passed` may have been derived from `FinalScore`; therefore, using it may cause data leakage.
* A high correlation alone is not sufficient. The relationship between the column and the problem should also make logical sense.








