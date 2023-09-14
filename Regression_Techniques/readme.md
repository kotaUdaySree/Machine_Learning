# Regression Models with Python

## Overview

This repository contains Python scripts and datasets for implementing various regression techniques. The models covered are:

### Simple Linear Regression

This model aims to estimate new salaries for new hire employees based on years of experience using simple linear regression techniques. The dataset (`Simple-Linear-Dataset.csv`) contains two columns: `Years_of_Expertise` and `Salary`.

### Multiple Linear Regression

This model delves into understanding factors affecting business profitability. The dataset (`Multiple-Linear-Dataset.csv`) includes columns for different products (`Product_1`, `Product_2`, `Product_3`), `Location`, and `Profit`. It explores the impact of product investments and location on maximizing profit.

### Polynomial Regression

The goal here is to estimate new salaries for new hire employees based on levels using polynomial regression. The dataset (`Polynomial-Dataset.csv`) contains columns for `Position`, `Level`, and `Salary`. This technique helps in finding the estimated salary for a new hire employee at level 6.5.

## Usage

### Simple Linear Regression

1. Import the necessary libraries.
2. Load the dataset.
3. Split the dataset into training and testing sets.
4. Fit a Simple Linear Regression model to the training set.
5. Predict the test set results.
6. Visualize the results.

### Multiple Linear Regression

1. Import the necessary libraries.
2. Load the dataset.
3. Encode categorical data if applicable.
4. Avoid the Dummy Variable Trap.
5. Split the dataset into training and testing sets.
6. Fit a Multiple Linear Regression model to the training set.
7. Predict the test set results.
8. Build the optimal model using Backward Elimination.

### Polynomial Regression

1. Import the necessary libraries.
2. Load the dataset.
3. Prepare the features (`X`) and target (`y`) variables.
4. Fit a Linear Regression model to the dataset (missing line).
5. Fit a Polynomial Regression model to the dataset with different degrees.
6. Visualize the results with different polynomial degrees.


This summary provides an overview of the models covered in repository along with a brief guide on how to use them. It also emphasizes the customization required based on specific datasets and problems.
