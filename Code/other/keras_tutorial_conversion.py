#!/usr/bin/env python3

""" Attempt to apply Keras tutorial to my project """

__appname__ = 'keras_tutorial_conversion.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


#################
#### Imports ####
#################

import matplotlib.pyplot as plt # for visual inspection
from keras.utils import to_categorical # for one-hot-encoding
from keras.models import Sequential # allows layer-by-layer model building
from keras.layers import Dense, Conv2D, Flatten # additional layers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras import regularizers, optimizers
import matplotlib.image as mpimg
import numpy as np
from os import listdir
from matplotlib import image
import random
from sklearn.model_selection import train_test_split
from PIL import Image


# Import data
cut_empty_spectra = list()
for filename in listdir('../Data/Spectrograms/Cut_Emptys/'):

	image = Image.open('../Data/Spectrograms/Cut_Emptys/' + filename)#.convert("L")
	new_img = image.resize((80,60))
	img_data = np.array(new_img)

	cut_empty_spectra.append(img_data)
	print('> loaded %s %s' % (filename, img_data.shape)) #1185


cut_gun_spectra = list()
for filename in listdir('../Data/Spectrograms/Cut_Gunshots/'):

	imgage = Image.open('../Data/Spectrograms/Cut_Gunshots/' + filename)#.convert("L")
	new_img = image.resize((80,60))
	img_data = np.array(new_img)

	cut_gun_spectra.append(img_data)
	print('> loaded %s %s' % (filename, img_data.shape)) #1185


# Take sample of empty spectrograms
indices = range(0, len(cut_empty_spectra))
random.seed(193)
sample = np.random.choice(indices, len(cut_gun_spectra))
cut_empty_spectra = [cut_empty_spectra[index] for index in sample]

# Make y data
y_gun = np.ones(len(cut_empty_spectra))
y_empty = np.zeros(len(cut_gun_spectra))

# Combine data
X_train = cut_empty_spectra + cut_gun_spectra
X_train = np.asarray(X_train)
Y_train = np.concatenate((y_empty, y_gun))

# Split into training and test data, 80:20
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=193)

# One-hot encode target data and reshape x
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

X_train = X_train.reshape(len(X_train),60,80,3)
X_test = X_test.reshape(len(X_test),60,80,3)


### Building the model ###
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(60,80,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)






