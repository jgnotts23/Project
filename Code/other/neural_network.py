#!/usr/bin/env python3

""" Project first attempt """

__appname__ = 'neural_network.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import numpy as np
from matplotlib import pyplot as plt

# Define features and labels
feature_set = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
labels = np.array([[1, 0, 0, 1, 1]])  # answers we are trying to predict
labels = labels.reshape(5, 1)  # transpose the matrix

# Define hyperparameters
np.random.seed(42)  # control random numbers
weights = np.random.rand(3, 1)  # assign random weights
bias = np.random.rand(1)  # assign random bias
lr = 0.05  # assign learning rate

# Define sigmoid (activation) function


def sigmoid(x):
    return 1/(1+np.exp(-x))

# Calculate derivative of sigmoid function


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#### Step 1 - Feed Forward ####
# First the dot product of the input feature matrix and weight matrix
# must be calculated (add bias) then fed into the activation function

#### Step 2 - Back Propagation ####
# This stage compares the prediction with the actual output and adjusts
# weights and bias accordingly, the 'training' phase


###
for epoch in range(20000):  # Define no. epochs (no. times to train algorithm)
    inputs = feature_set  # store features as inputs

    # feedforward step1 - find dot product of input and weights
    XW = np.dot(feature_set, weights) + bias

    # feedforward step2 - pass through activation function
    z = sigmoid(XW)

    # backpropagation step 1 - calculate error
    error = z - labels

    print(error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)  # update weights

    for num in z_delta:  # update bias values
        bias -= lr * num


#### Testing ###
single_point = np.array([1, 0, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)
