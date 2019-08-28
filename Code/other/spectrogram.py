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
from pydub import AudioSegment
import glob
import librosa
import librosa.display
import pandas as pd
from subprocess import call
 

test = pd.DataFrame(columns = ('ID', 'Class', 'File'))
train = pd.DataFrame(columns = ('ID', 'Class', 'File'))


# Make spectrograms for cut gunshot data
files = []
i = 0
for root, dirs, files in os.walk("../Data/Gunshot_recordings/Cut_Gunshot_Recordings/"):
    for file in files:
        if file.endswith((".WAV", ".wav")):
            name = (os.path.splitext(file)[0])
            #print(root)
            print(name)
            i = i + 1
            plot.interactive(False)
            clip, sample_rate = librosa.load(root + '/' + file, sr=None)
            fig = plot.figure(figsize=[0.72,0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

            if i % 4 == 0:
                filename  = '../Data/Spectrograms/Test/' + str(i) + '.jpg'
                test.at[i, 'ID'] = i
                test.at[i, 'Class'] = 1
                test.at[i, 'File'] = name
            else:
                filename  = '../Data/Spectrograms/Train/' + str(i) + '.jpg'
                train.at[i, 'ID'] = i
                train.at[i, 'Class'] = 1
                train.at[i, 'File'] = name

            plot.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
            plot.close()    
            fig.clf()
            plot.close(fig)
            plot.close('all')
            del filename,name,clip,sample_rate,fig,ax,S


# Make spectrograms for cut empty data
files = []
for root, dirs, files in os.walk("../Data/Gunshot_recordings/Cut_Empty_Recordings/"):
    for file in files:
        if file.endswith((".WAV", ".wav")):
            name = (os.path.splitext(file)[0])
            #print(root)
            print(name)
            i = i + 1
            plot.interactive(False)
            clip, sample_rate = librosa.load(root + '/' + file, sr=None)
            fig = plot.figure(figsize=[0.72,0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

            if i % 4 == 0:
                filename  = '../Data/Spectrograms/Test/' + str(i) + '.jpg'
                test.at[i, 'ID'] = i
                test.at[i, 'Class'] = 0
                test.at[i, 'File'] = name
            else:
                filename  = '../Data/Spectrograms/Train/' + str(i) + '.jpg'
                train.at[i, 'ID'] = i
                train.at[i, 'Class'] = 0
                train.at[i, 'File'] = name

            plot.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
            plot.close()    
            fig.clf()
            plot.close(fig)
            plot.close('all')
            del filename,name,clip,sample_rate,fig,ax,S



test.to_csv('../Data/Spectrograms/Test/test.csv', index=False)
train.to_csv('../Data/Spectrograms/Train/train.csv', index=False)



## Cut empty recordings into 4 second chunks ##
files = []
four_seconds = 4 * 1000 # pydub works in milliseconds
for root, dirs, files in os.walk("../Data/Gunshot_recordings/Uncut_Empty_Recordings/Uncut_Empty_Recordings3/"):
        for file in files:
            if file.endswith(".WAV"):
                folder = os.path.basename(root)
                basename = (os.path.splitext(file)[0])
                uncut_wav = AudioSegment.from_wav(root + '/' + file)

                for i in range(0, 15):
                    clip = uncut_wav[(i*four_seconds):(four_seconds) * (i+1)]
                    new_path = "../Data/Gunshot_recordings/Cut_Empty_Recordings/Cut_Empty_Recordings3/" + folder + '/' + basename + '_' + str(i) + '.WAV'
                    clip.export(new_path, format="wav")
                    
                    # samplingFrequency, signalData = wavfile.read(new_path)
                    # plot.subplot(111)
                    # plot.specgram(signalData,Fs=samplingFrequency) 
                    # plot.xlabel('Time')
                    # plot.ylabel('Frequency')
                    # plot.savefig('../Data/Spectrograms/Cut_Emptys/' + basename + '_' + str(i) + '.jpg')
                    
                print(file + ' done!')
