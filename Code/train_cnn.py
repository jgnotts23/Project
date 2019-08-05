#!/usr/bin/env python3

"""  """

__appname__ = 'train_cnn.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


#################
#### Imports ####
#################
import os
import pandas as pd
from glob import glob
import numpy as np
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import random
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from PIL import Image
import sys


# Function to add file extenstion to a string
def append_ext(fn):
    return str(fn) + ".jpg"

# Read in test and train data
gunshot_data = pd.read_csv('../Data/Gunshot_recordings/Spectra/gun_results.csv', dtype=str)
blank_data = pd.read_csv('../Data/Gunshot_recordings/Spectra/results.csv', dtype=str)
blank_data = blank_data.sample(n=358, random_state=1)
traindf = gunshot_data.append(blank_data)
train_dir = '../Data/Gunshot_recordings/Spectra'

#testdf = pd.read_csv(str(sys.argv[1]),dtype=str)
#test_dir = str(sys.argv[2])

# Add file extensions to ID
traindf["ID"]=traindf["ID"].apply(append_ext)

# Rescale normalises images, helps improve convergence
datagen = ImageDataGenerator(rescale = 1./255., validation_split = 0.25)

# Training data generator, generates batches of
# augmented/normalised data
train_generator = datagen.flow_from_dataframe(
    dataframe = traindf, #use training dataset
    directory = train_dir,
    x_col = "ID", #specify image name
    y_col = "Class", #specify where category is (0 or 1)
    subset = "training",
    batch_size = 1, #should divide by total no. of train + valid data
    seed = 42,
    shuffle = True,
    class_mode = "categorical",
    target_size = (64,64)) #resize

# Validation data generator
valid_generator = datagen.flow_from_dataframe(
    dataframe = traindf,
    directory = train_dir,
    x_col = "ID",
    y_col = "Class",
    subset = "validation",
    batch_size = 1,
    seed = 42,
    shuffle = True,
    class_mode = "categorical",
    target_size = (64,64))


### Model definition ###
model = Sequential() #linear stack of layers

# Layer that creates a convolutional kernel that is convolved
# with the layer input to produce a tensor of outputs
model.add(Conv2D(32, (3, 3), # no. filters, kernel size
                 input_shape=(64,64,3)))
# Layer to apply activation function
model.add(Activation('relu'))
# Pooling layer to reduce spacial size of the image
# representation, reducing no. params and computation
model.add(MaxPooling2D(pool_size=(2, 2)))

# Repeat - Convolute, activate, pool
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Repeat again
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer converts data to a 1D linear vector
model.add(Flatten())
# Regular layer
model.add(Dense(64))
model.add(Activation('relu'))
# Dropout layer randomly drops out nodes during training
# in an effort to reduce overfitting
model.add(Dropout(0.5))
# Final classification
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile phase
# lr = learning rate
# decay = learning rate decay after each update
model.compile(optimizers.rmsprop(lr = 0.0005, decay = 1e-6), loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()


#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator = train_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = valid_generator,
                    validation_steps = STEP_SIZE_VALID,
                    epochs = 100
)
model.evaluate_generator(generator = valid_generator, steps = STEP_SIZE_VALID
)

model.save('../Data/my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
