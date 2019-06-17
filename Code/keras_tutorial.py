#!/usr/bin/env python3

""" Convolutional neural networks tutorial with 
Keras """

__appname__ = 'keras_tutorial.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


#################
#### Imports ####
#################

from keras.datasets import mnist # example dataset
import matplotlib.pyplot as plt # for visual inspection
from keras.utils import to_categorical # for one-hot-encoding
from keras.models import Sequential # allows layer-by-layer model building
from keras.layers import Dense, Conv2D, Flatten # additional layers

# Download mnist data and split into train and test sets
# X_train and X_test contain images (handwritten numbers)
# y_train and y_test contain digits that the images represent (correct answer)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0


###################################
#### Exploratory data analysis ####
###################################

# Lets plot the first image in the dataset
plt.imshow(X_train[0]) #handwritten 5

#check image shape
X_train[0].shape # 28x28 matrix of pixel values


#############################
#### Data pre-processing ####
#############################

# Now need to reshape data inputs, X_train and X_test, to fit model
# .reshape(num_images, shape, shape, 1 = greyscale)
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# Need to 'one-hot-encode' target variable
# A column will be created for each output category and
# a binary variable is inputted for each category
# e.g. for 5 in this dataset you'd get:
# array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

# One-hot encode target (y) column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]


############################
#### Building the model ####
############################

# Create model
model = Sequential() # empty model, no layers yet

# Add model layers with .add()
# First 2 layers are Conv2D layers, these are convolution
# layers that will deal with input images, which are
# seen as 2-dimensional matrices. 64 and 32 correspond
# to the number of nodes in each layer, can be adjusted
# for different datasets
# kernel_size is the size of the filter matrix for our
# convolution, i.e. 3 = a 3x3 filter matrix. Can be
# thought of as 'feature extractors'
# Activation function used is the Rectified Linear
# Activation (ReLU). The first layer takes an input shape
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))

# Flatten layer serves as a connection between the convolution
# and dense layers
model.add(Flatten())

# A 'Dense' layer type will be used as the output layer.
# It has 10 nodes, one for each possible outcome (0-9)
# This layer has a 'softmax' activation function. This
# makes the output sum to 1 so it can be interpreted as
# probabilities and the prediction is based on the highest prob.
model.add(Dense(10, activation='softmax'))


#############################
#### Compiling the model ####
#############################

# Compiling takes 3 parameters:
# Optimizer, loss and metrics
# Optimizer controls learning rate. In this case we are
# using 'Adam' which adjust learning rate throughout training
# We will use 'categorical_crossentropy' for our loss
# function. A lower score indicates better model performance
# We will use the 'accuracy' metric to see the accuracy score
# on the validation (test) set whenAudioMoth: Evaluation of a smart open acoustic device for monitoring biodiversity and the environment we train the model

# Compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


############################
#### Training the model ####
############################

# We will use the fit() function to train the model with
# the training data, target data, validation data, and
# number of epochs as parameters 
# epoch = single cycle through data

# Train the model
# training data = X_train
# target data = y_train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# After 3 epochs, you should have about 97.57% accuracy 
# on the validation set.
# This is now a fully-functioning CNN!!


### Using model to make predictions ###

# To see the model predictions for the test data, we can
# use the .predict() function which returns an array of
# 10 numbers, each corresponding to the probability that 
# the input image represents each digit (0-9)

# Predict first 4 images in the test set
model.predict(X_test[:4])

# We can see the model predicted 7, 2, 1 and 0 for the
# first four images. 

# Actual results for first 4 images in test set
y_test[:4] # Correct!




