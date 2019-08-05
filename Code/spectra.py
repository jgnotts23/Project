#!/usr/bin/env python3

""" """

__appname__ = 'spectra.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import matplotlib.pyplot as plot
#from scipy.io import wavfile
import numpy as np
import os
from pydub import AudioSegment
#import glob
import librosa
import librosa.display
import pandas as pd
#import random
import subprocess
from subprocess import call
import sys

np.random.seed(23)

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
#train = pd.DataFrame(columns=('ID', 'Class', 'File'))


#### Make spectrograms ###
def make_spectra(source, target):

    files = []
    i = 0

    #cmd = 'ls -1 ' + source + ' | wc -l > count.txt'
    #number_files = subprocess.call(cmd, shell=True)

    #number_files = int(open("count.txt").readline().rstrip())

    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith((".WAV", ".wav")):
        #while True:
            #try:
                name = (os.path.splitext(file)[0])
                plot.interactive(False)

                clip, sample_rate = librosa.load(root + '/' + file, sr=None)

            #except EOFError as error:
                #print("EOFError on " + str(file))
                #continue

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
                test.at[i, 'Class'] = '1'
                test.at[i, 'File'] = name

                plot.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
                plot.close()
                fig.clf()
                plot.close(fig)
                plot.close('all')
                #
                # progress = int((i / total) * 100)
                # print(str(progress) + '%')
                #print(str(i) + '/' + str(number_files))

                i = i + 1
                del filename, name, clip, sample_rate, fig, ax, S

#x = int(os.getenv("PBS_ARRAY_INDEX")) #assigned by HPC shell script
#folder = '/Audio ' + str(x)

# total=0
# x=[]
# for file in os.listdir(str(sys.argv[1])):
#     if file.endswith('.wav'):
#         x.append(file)
#         total+=1
# print('Audio files found: ' +str(total))

#count = len([name for name in os.listdir(sys.argv[1]) if os.path.isfile(name)])
#print(str(count) + ' audio files found')

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
make_spectra(str(sys.argv[1]), str(sys.argv[2]))
test.to_csv(str(sys.argv[2]) + '/gun_results.csv', index=False)

#run spectra.py /media/jacob/Samsung_external/Rancho_bajo/Cut /media/jacob/Samsung_external/Rancho_bajo/Spectra
