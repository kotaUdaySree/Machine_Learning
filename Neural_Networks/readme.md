# Stock Price Prediction with Neural Networks

This repository contains Python code for predicting stock prices using artificial neural networks. The implementation includes features for data preprocessing, feature engineering, and training a feedforward neural network. Below is a brief overview of the key components:

## Code Summary:

1. **Feature Generation and Extraction**:
   - Functions are provided to generate additional features from the original financial dataset, such as moving averages, ratios, standard deviations, and returns.

2. **Dataset Splitting**:
   - The data is divided into training and testing sets based on specified date ranges. The training set spans from 1988 to 2018, while the testing set covers the year 2019.

3. **Feature Scaling**:
   - Standardization of features is performed to improve the convergence of the neural network during training.

4. **Neural Network Training**:
   - A feedforward neural network with two hidden layers (16 and 8 units) is constructed. The ReLU activation function is employed, and training is carried out using the scaled training data.

5. **Making Predictions**:
   - The trained neural network is used to make predictions on the scaled test data.

6. **Evaluation**:
   - The script calculates and prints the R-squared score, mean absolute error (MAE), and mean squared error (MSE) to assess the model's performance on the test data.

7. **Plotting Results**:
   - The results are visualized with a plot showing true closing prices in red and predicted prices in green for the test set.

8. **Output**:
   - The output includes the predicted values for each data point in the test set.

## Usage:

1. **Dataset**:
   - Ensure you have a financial dataset in CSV format, containing columns like 'Open', 'High', 'Low', 'Close', and 'Volume' for different dates.

2. **Dependencies**:
   - Make sure you have the required Python libraries installed, including Pandas, NumPy, Scikit-learn, and Matplotlib.

3. **Running the Code**:
   - Execute the Python script, and it will perform feature generation, model training, and prediction. The results will be displayed, and a plot will be generated.




