#!/usr/bin/env python3

""" """

__appname__ = 'spectra.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import matplotlib.pyplot as plot
import numpy as np
import os
from pydub import AudioSegment
import librosa
import librosa.display
import pandas as pd
import subprocess
from subprocess import call
import sys

np.random.seed(23)

def make_spectra(source, target):

    files = []
    i = 0

    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith((".WAV", ".wav")):

                name = (os.path.splitext(file)[0]) # extract filename
                plot.interactive(False)

                # Load audio data
                clip, sample_rate = librosa.load(root + '/' + file, sr=None)

                # Plot spectrogram
                fig = plot.figure(figsize=[0.72, 0.72])
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
                librosa.display.specshow(
                    librosa.power_to_db(S, ref=np.max))

                filename = target + '/' + str(i) + '.jpg'
                test.at[i, 'ID'] = i
                test.at[i, 'Class'] = 'NA'
                test.at[i, 'File'] = name

                plot.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
                plot.close()
                fig.clf()
                plot.close(fig)
                plot.close('all')

                i = i + 1

                progress = int((i / total) * 100)
                print(str(progress) + '%')

                del filename, name, clip, sample_rate, fig, ax, S


total=0
x=[]
for file in os.listdir(str(sys.argv[1])):
    if file.endswith(('.wav', '.WAV')):
        x.append(file)
        total+=1
print('Audio files found: ' +str(total))

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
make_spectra(str(sys.argv[1]), str(sys.argv[2]))

def append_jpg(fn):
    return str(fn) + ".jpg"

def append_wav(fn):
    return str(fn) + ".wav"

test["ID"]=test["ID"].apply(append_jpg)
test["File"]=test["File"].apply(append_wav)



test.to_csv(str(sys.argv[2]) + '/spectra.csv', index=False)

#run spectra.py /media/jacob/Samsung_external/SQ283/Cut/Audio2 /media/jacob/Samsung_external/SQ283/Spectra/Audio2
