#!/usr/bin/env python3

""" Working on spectrograms """

__appname__ = 'spectrogram.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import matplotlib.pyplot as plot
from scipy.io import wavfile
import numpy as np
import os


def make_spectrogram(samplingFrequency, signalData, basename):

    #wavfile = '../Data/Cut_Gunshot_Recordings1/' + wavfile

    # Read in sample rate and data from .wav file
    #samplingFrequency, signalData = wavfile.read(wavfile)

    # Plot the signal read from wav file
    plot.subplot(211) # means 2rows, 1col, index 1
    plot.title('Spectrogram of ' + basename)
    plot.plot(signalData) # sample no. against amplitude
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')

    # Plot spectrogram
    plot.subplot(212) # 2rows, 1col, index 2
    # Fs = scalar, samples per unit time, used to calculate
    # the Fourier frequencies in cycles per unit time
    # Plots time against frequency with amplitude as
    # Colour intensity
    plot.specgram(signalData,Fs=samplingFrequency) 
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.savefig('../Results/' + basename + '.jpg')



# Run on all .wav files in directory
#import os
#import glob
#path = "../Data/Cut_Gunshot_Recordings/Cut_Gunshot_Recordings1/002/"
#files = os.listdir(path)
#zero = []

#for filename in glob.glob(os.path.join(path, '*.wav')):
    #print(filename)
    #base = os.path.basename(filename)
    #basename = (os.path.splitext(base)[0])
    #samplingFrequency, signalData = wavfile.read(filename)
    #make_spectrogram(samplingFrequency, signalData, basename)

files = []
for root, dirs, files in os.walk("../Data/Cut_Gunshot_Recordings/"):
    for file in files:
        if file.endswith(".wav"):
            print(file + ' done!')
            #base = os.path.basename(file) = file
            basename = (os.path.splitext(file)[0])
            samplingFrequency, signalData = wavfile.read(root + '/' + file)
            #make_spectrogram(samplingFrequency, signalData, basename)
            plot.subplot(111)
            plot.specgram(signalData,Fs=samplingFrequency) 
            plot.xlabel('Time')
            plot.ylabel('Frequency')
            plot.savefig('../Results/' + basename + '.jpg')
            #print(os.path.join(root, file))