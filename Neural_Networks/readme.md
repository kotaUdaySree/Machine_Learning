# Stock Price Prediction with Neural Networks

Predicting stock prices is a challenging yet crucial task for investors and financial analysts. This repository provides a Python-based solution that employs artificial neural networks to predict stock prices. This README file provides a comprehensive overview of the project, its components, and how to use it effectively.

## Table of Contents

- [Introduction](#introduction)
- [Code Summary](#code-summary)
  - [Feature Generation and Extraction](#feature-generation-and-extraction)
  - [Dataset Splitting](#dataset-splitting)
  - [Feature Scaling](#feature-scaling)
  - [Neural Network Training](#neural-network-training)
  - [Making Predictions](#making-predictions)
  - [Evaluation](#evaluation)
  - [Plotting Results](#plotting-results)
- [Usage](#usage)
  - [Dataset](#dataset)
  - [Dependencies](#dependencies)
  - [Running the Code](#running-the-code)
- [Note](#note)

## Introduction

Predicting stock prices is a complex problem due to the myriad of factors influencing financial markets. Artificial neural networks, particularly feedforward neural networks, have shown promise in capturing intricate patterns within financial time series data. This project leverages neural networks to predict stock prices based on historical data and relevant features.

## Code Summary

### Feature Generation and Extraction

The code includes functions to generate and extract additional features from the original financial dataset. These features go beyond raw price and volume data and encompass moving averages, ratios, standard deviations, and returns. These features aim to capture relevant information that can improve prediction accuracy.

### Dataset Splitting

The dataset is split into two subsets: the training set and the testing set. The training data spans from 1988 to 2018, while the testing data covers the year 2019. Proper dataset splitting ensures that the model is evaluated on unseen data, which is crucial for assessing its real-world predictive performance.

### Feature Scaling

Before feeding the data into the neural network, feature scaling is applied using the `StandardScaler` from `sklearn.preprocessing`. Scaling the features to have a consistent scale helps in the convergence of the neural network during training.

### Neural Network Training

A feedforward neural network architecture is employed for stock price prediction. This network consists of two hidden layers with 16 and 8 units, respectively, and ReLU (Rectified Linear Unit) activation functions. The model is trained using the scaled training data. The choice of architecture and activation functions can be further customized based on specific requirements.

### Making Predictions

The trained neural network is utilized to make predictions on the scaled test data. Predictions are generated for each data point in the test set.

### Evaluation

To assess the model's performance, key evaluation metrics are computed, including:
- R-squared score: Measures the proportion of the variance in the dependent variable (stock prices) that is predictable from the independent variables (features).
- Mean Absolute Error (MAE): Represents the average absolute difference between predicted and actual prices.
- Mean Squared Error (MSE): Quantifies the average squared difference between predicted and actual prices.

These metrics provide insights into the accuracy and reliability of the model's predictions.

### Plotting Results

The project also includes functionality to visualize the model's predictions. A plot is generated to display both the true closing prices (in red) and the predicted prices (in green) for the test set. Visualization aids in understanding how well the model captures price trends.

## Usage

To utilize this stock price prediction project, follow these steps:

### Dataset

Download the financial dataset in CSV format. The dataset should include relevant columns such as 'Open,' 'High,' 'Low,' 'Close,' and 'Volume' for different dates. The quality and completeness of your dataset significantly impact the model's performance.

### Dependencies

Make sure you have the necessary Python libraries installed to run the code successfully. Key libraries used in this project include:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

You can typically install these libraries using pip or conda.

### Running the Code

Execute the provided Python script to initiate the stock price prediction process. The script performs feature generation, neural network training, and prediction. The results, including evaluation metrics and a visual plot, will be displayed. Ensure that you address any warnings or issues related to the sigmoid function mentioned in the code.

