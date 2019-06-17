#!/usr/bin/env python3

""" Convolutional neural networks tutorial """

__appname__ = 'cnn_tutorial.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import sys

# Load and store train and test images
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

### Analyse the data ###
print('Training data shape : ', train_X.shape, train_Y.shape) #60,000 samples of 28x28 dimension
print('Testing data shape : ', test_X.shape, test_Y.shape) #10,000 samples, same dimensions

# Find unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses) #10 output classes
print('Output classes : ', classes) #classes range from 0 to 9

### Visualise data ###
# Making greyscale images with pixel values in range 0 to 255
plt.figure(figsize=[5,5])

# Display first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

### Data preprocessing ###
# Convert each image into a matix of size 28x28x1
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X.shape, test_X.shape

# Convert from int8 format to float32 format
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# Rescale between 0 and 1
train_X = train_X / 255
test_X = test_X / 255

# Convert to one-hot encoding vector from categorical
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# Split the data into training and test data
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size = 0.2, random_state = 13)
train_X.shape, valid_X.shape, train_label.shape, valid_label.shape


### Model the data ###
batch_size = 64
epochs = 20
num_classes = 10

# Make new model
fashion_model = Sequential()
# Add first convolutional layer with Conv2D (because images)
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
# Add LeakyReLU activation function, helps network learn non-linear decision boundaries
# Also helps fit dying Rectified Linear Units (ReLUs)
fashion_model.add(LeakyReLU(alpha=0.1))
# Add max-pooling layer
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
# Repeat
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
# Dense layer with softmax activation function with 10 units                  
fashion_model.add(Dense(num_classes, activation='softmax'))

### Compile the model ###
# Using Adam optimiser - very popular
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# Visualise the layers
fashion_model.summary()

### Train the model ###
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))



### Model evaluation ###
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

plt.close()
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()