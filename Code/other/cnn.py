#!/usr/bin/env python3

""" First full attempt at creating a CNN """

__appname__ = 'cnn.py'
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
positive_df = pd.read_csv('/media/jacob/Samsung_external/Test/results_guns.csv',dtype=str)
negative_df = pd.read_csv('/media/jacob/Samsung_external/Test/results_blank.csv',dtype=str)
traindf = positive_df.append(negative_df)
#traindf = pd.read_csv(sys.argv[1],dtype=str) #training
#testdf = pd.read_csv(sys.argv[2],dtype=str) #test

train_dir = '/media/jacob/Samsung_external/Test/Merged_Spectra'

# Add file extensions to ID
traindf["ID"]=traindf["ID"].apply(append_ext)
#testdf["ID"]=testdf["ID"].apply(append_ext)

# example image
#image = Image.open('../Data/Spectrograms/Train/1.jpg')#.convert("L")
#img_data = np.array(image)


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
                    epochs = 20
)
model.evaluate_generator(generator = valid_generator, steps = STEP_SIZE_VALID
)


## Test time ##
# Make list of test file names
test_IDs = []
for file in os.listdir('/media/jacob/Samsung_external/SQ258/Spectra/Audio2'):
    if file.endswith(".jpg"):
        test_IDs.append(file)

#test_filenames = os.listdir(sys.argv[3])
spectra_df = pd.read_csv('/media/jacob/Samsung_external/SQ258/Spectra/Audio2/results.csv',dtype=str)
test_filenames = np.asarray(spectra_df['File'])

# Make dataframe for test data
testdf=pd.DataFrame({"ID":test_IDs,
                      "Predicted_class":"",
                      "File":test_filenames})

test_dir = '/media/jacob/Samsung_external/SQ258/Spectra/Audio2'

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=test_dir, #path to directory that contains the images
    x_col="ID", #column with image filenames
    y_col=None,
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

#Fetch labels from train gen for testing
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
#print(predictions[0:6])

#zeros = np.zeros(len(predictions))



test_IDs=test_generator.filenames
results=pd.DataFrame({"ID":test_IDs,
                      "Predicted_class":predictions,
                      "File":test_filenames})

positives = results.loc[results['Predicted_class'] == 1]


# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})


# i = 0
# for i in range(0, len(predictions)):
#     filename = results['Filename'][i]
#     actual = testdf.loc[testdf['ID'] == filename, 'Class'].copy()
#     results['Actual'][i] = actual
#     i = i + 1

results.to_csv("../Data/results.csv",index=False)

#Predictions = np.array(results['Predictions'].astype(int))
#Actual = np.array(results['Actual'].astype(int))

# def test_accuracy(Predictions, Actual):
#
#
#     x = np.equal(Predictions, Actual)
#     counts = np.unique(x, return_counts=True)
#
#     score = counts[1][1] / (counts[1][1] + counts[1][0])
#
#     return score
#
# score = test_accuracy(Predictions, Actual)
# score







#
# testdf = pd.read_csv('/media/jgnotts23/Samsung_external/Results/test.csv',dtype=str) #test
# testdf["ID"]=testdf["ID"].apply(append_ext)
#
# ## Test time ##
# test_datagen=ImageDataGenerator(rescale=1./255.)
# test_generator=test_datagen.flow_from_dataframe(
#     dataframe=testdf,
#     directory="/media/jgnotts23/Samsung_external/Results/", #path to directory that contains the images
#     x_col="ID", #column with image filenames
#     y_col=None,
#     batch_size=5,
#     seed=42,
#     shuffle=False,
#     class_mode=None,
#     target_size=(64,64))
# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
#
#
# test_generator.reset()
# pred=model.predict_generator(test_generator,
# steps=STEP_SIZE_TEST,
# verbose=1)
# predicted_class_indices=np.argmax(pred,axis=1)
#
# #Fetch labels from train gen for testing
# labels = (train_generator.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]
# print(predictions[0:6])
#
# zeros = np.zeros(len(predictions))
#
# filenames=test_generator.filenames
# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions,
#                       "Actual":zeros})
#
#
# i = 0
# for i in range(0, len(predictions)):
#     filename = results['Filename'][i]
#     actual = testdf.loc[testdf['ID'] == filename, 'Class'].copy()
#     results['Actual'][i] = actual
#     i = i + 1
#
# results.to_csv("/media/jgnotts23/Samsung_external/results.csv",index=False)
#
# positive = results.loc[results['Predictions'] == '1']
# positive = np.asarray(positive['Filename'])
# first = testdf.loc[testdf['ID'].isin(positive)]
#
#
# Predictions = np.array(results['Predictions'].astype(int))
# Actual = np.array(results['Actual'].astype(int))
