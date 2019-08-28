#!/usr/bin/env python3

"""  """

__appname__ = 'test_cnn.py'
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
from keras.models import load_model
from keras import regularizers, optimizers
from PIL import Image
import sys
from scipy import stats


model = load_model('../Data/Updated_model_extra_negs.h5')


#test_filenames = os.listdir(sys.argv[3])
#positives = pd.read_csv(str(sys.argv[1]),dtype=str, header=None, names=["ID", "Class", "File"])
#negatives = pd.read_csv(str(sys.argv[2]),dtype=str, header=None, names=["ID", "Class", "File"])




#columnsTitles=["ID", "Class", "File"]
#negatives=negatives.reindex(columns=columnsTitles)


testdf = pd.read_csv(str(sys.argv[1]),dtype=str)
#testdf = pd.concat([positives, negatives])
test_filenames = np.asarray(testdf['File'])

testdf['ID'] = [str(sys.argv[3]) + s for s in testdf['ID']]


#def append_ext(fn):
    #return str(fn) + ".jpg"

#testdf["ID"]=testdf["ID"].apply(append_ext)


# Make dataframe for test data
#testdf=pd.DataFrame({"ID":test_IDs,
                      #"Predicted_class":"",
                      #"File":test_filenames})

test_dir = str(sys.argv[2])

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
#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]
#print(predictions[0:6])

#zeros = np.zeros(len(predictions))



test_IDs=test_generator.filenames
results=pd.DataFrame({"ID":test_IDs,
                      "Predicted_class":predicted_class_indices,
                      "File":test_filenames})

#def append_wav(fn):
#    return str(fn) + ".wav"

#results["File"]=results["File"].apply(append_wav)

positives = results.loc[results['Predicted_class'] == 1]
#positive_filenames = np.asarray(positives['File'])

positives.to_csv(str(sys.argv[2]) + '/extranegs_model.csv', index=False)

#np.savetxt(str(sys.argv[2]) + '/positives.csv', positive_filenames, delimiter=",", fmt='%s')
#cat ../Spectra/positives.csv | xargs -I % cp % positives

#run test_cnn.py /media/jacob/Samsung_external/SQ283/Spectra/Audio1/confirmed.csv /media/jacob/Samsung_external/SQ283/Spectra/Audio1/negatives.csv /media/jacob/Samsung_external/SQ283/Spectra/Audio1

#run test_cnn.py /media/jacob/Samsung_external/SQ258/Spectra/Audio1/gun_results.csv /media/jacob/Samsung_external/SQ258/Spectra/Audio1 SQ258_


# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})


# i = 0
# for i in range(0, len(predictions)):
#     filename = results['Filename'][i]
#     actual = testdf.loc[testdf['ID'] == filename, 'Class'].copy()
#     results['Actual'][i] = actual
#     i = i + 1

#results.to_csv("../Data/results.csv",index=False)
