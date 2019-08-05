#!/usr/bin/env python3

""" Splits wav files """

__appname__ = 'split_wav.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
#import matplotlib.pyplot as plot
#from scipy.io import wavfile
import numpy as np
import os
from pydub import AudioSegment
#import glob
#import librosa
#import librosa.display
import pandas as pd
#import random
import subprocess
from subprocess import call
import sys


def split_wav(source, target, length):

    """ Splits a directory of audio file into chunks of specified length (seconds)
    and saves the clips in a specified folder """

    #cmd = 'ls -1 ' + source + ' | wc -l > count.txt'
    #number_files = subprocess.call(cmd, shell=True)

    #number_files = int(open("count.txt").readline().rstrip())
    #sample_size = 10

    iteration = 1
    clip_length = length * 1000  # pydub works in milliseconds
    files = []

    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith((".WAV", ".wav")):
                #print(str(iteration) + '/' + str(number_files))
                #if iteration in sample:
                #if os.path.getsize(root + '/' + file):

                basename = (os.path.splitext(file)[0])
                uncut_wav = AudioSegment.from_wav(root + '/' + file)
                duration = uncut_wav.duration_seconds
                no_clips = int(duration // (clip_length / 1000))

                i = 0
                for i in range(0, no_clips):
                    clip = uncut_wav[(i*clip_length):(clip_length) * (i+1)]
                    new_path = target + '/' + basename + '_' + str(i) + '.wav'
                    clip.export(new_path, format="wav")
                    i = i + 1

                iteration = iteration + 1

                progress = int((iteration / total) * 100)
                print(str(progress) + '%')

                #else:
                    #continue

                #else:
                   # iteration = iteration + 1
                   # continue


#x = int(os.getenv("PBS_ARRAY_INDEX")) #assigned by HPC shell script
#folder = '/Audio ' + str(x)

source = str(sys.argv[1]) #+ folder
target = str(sys.argv[2]) #+ folder

total=0
x=[]
for file in os.listdir(str(sys.argv[1])):
    if file.endswith(('.wav', '.WAV')):
        x.append(file)
        total+=1
print('Audio files found: ' +str(total))

#target = str(sys.argv[2])
length = 4

# Create directory
#dirName = source + '/Cut'
#os.mkdir(dirName)

#run split_wav.py /media/jacob/Samsung_external/Rancho_bajo/Audio1 /media/jacob/Samsung_external/Rancho_bajo/Cut

split_wav(source, target, length)
