#!/usr/bin/env python3

""" First attempt at CNN """

__appname__ = 'cnn_firsttry.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import numpy as np
from PIL import Image
import os
from scipy import ndimage
from scipy import misc
from scipy import stats
import matplotlib.pyplot as plt
import random


# Make list of spectrogram image files
spectro_names = []
for file in os.listdir("../Results/"):
    if file.endswith(".jpg"):
        spectro_names.append(file)

# Read in spectrograms
spectrograms = []
for file in spectro_names:
    jpgfile = Image.open("../Results/" + file)#.convert('L') #converts to greyscale
    spectrograms.append(jpgfile)
    #print(jpgfile.bits, jpgfile.size, jpgfile.format)

# Convert images to numpy arrays
i = 0
spectro_data = []
for image in spectrograms:
    f = spectrograms[i]
    f = np.array(f.getdata())
    spectro_data.append(f)
    i = i + 1

# Split into training and test data 80:20
random.seed(80) # to ensure split is always the same
train_X_indices = random.sample(range(194), 154)
train_X = np.split(spectro_data, train_X_indices)

