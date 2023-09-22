# Movie Review Sentiment Analysis

This repository contains code for performing sentiment analysis on movie reviews using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The analysis is performed on the IMDb dataset, which consists of positive and negative movie reviews.

## Table of Contents

- [1. Analyzing and preprocessing the movie review data](#1-analyzing-and-preprocessing-the-movie-review-data)
- [2. Developing a simple LSTM network](#2-developing-a-simple-lstm-network)
- [3. Boosting the performance with multiple LSTM layers](#3-boosting-the-performance-with-multiple-lstm-layers)
- [4. Evaluating the Model](#4-evaluating-the-model)
- [5. Plotting the Learning Curve](#5-plotting-the-learning-curve)

## 1. Analyzing and Preprocessing the Movie Review Data

In this section, we prepare the data for training:

### Importing Necessary Modules

We import the required libraries, including TensorFlow, Matplotlib, and scikit-learn modules for data handling and visualization.

```python
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
```

### Loading and Preprocessing the Data

Next, we load and preprocess the IMDb dataset. We set the vocabulary size and load the training and testing data.

```python
vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
```

### Exploring the Data

We print out information about the loaded data, such as the number of samples, the number of positive and negative samples, and the number of test samples.

```python
print('Number of training samples: ', len(y_train))
print('Number of positive samples:', sum(y_train))
print('Number of test samples:', len(y_test))
```

### Sample Representation

We print out the representation of a sample. Each word is represented by an integer, indicating the frequency of the word.

```python
print(X_train[0])
```

### Mapping Integer to Word

We create a word dictionary to map integers back to words.

```python
word_index = imdb.get_word_index()
index_word = {index: word for word, index in word_index.items()}
```

### Analyzing Length of Reviews

We analyze the length of each review to understand the distribution.

```python
review_lengths = [len(x) for x in X_train]
plt.hist(review_lengths, bins=10)
plt.show()
```

### Padding Sequences

To ensure uniform length for input sequences, we pad shorter reviews with zeros and truncate longer reviews.

```python
maxlen = 200
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
```

## 2. Developing a Simple LSTM Network

In this section, we build a basic LSTM network for sentiment analysis:

### Initializing the Model

We set a random seed and initiate a sequential model.

```python
tf.random.set_seed(42)
model = models.Sequential()
```

### Adding Embedding Layer

We add an embedding layer to convert input sequences into dense vectors.

```python
embedding_size = 32
model.add(layers.Embedding(vocab_size, embedding_size))
```

### Adding LSTM and Output Layers

We incorporate a single LSTM layer with 50 nodes and a dense output layer for binary classification.

```python
model.add(layers.LSTM(50))
model.add(layers.Dense(1, activation='sigmoid'))
```

### Compiling the Model

We compile the model with binary cross-entropy loss and the Adam optimizer.

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Training the Model

We train the model with batches of 64 size for three epochs.

```python
batch_size = 64
n_epoch = 3
model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, y_test))
```

### Evaluating the Model

We evaluate the classification accuracy of the model on the testing set.

```python
acc = model.evaluate(X_test, y_test, verbose=0)[1]
print('Test accuracy:', acc)
```

---

[Continue with sections 3, 4, and 5 in a similar manner]

...

## Usage

To run the code, make sure you have the required libraries installed. You can install them using the following command:
Then, you can execute each section of the code in a Python environment.

## Acknowledgements

- The IMDb dataset is part of the Keras library and can be found [here](https://keras.io/api/datasets/imdb/).
