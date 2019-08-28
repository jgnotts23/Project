#!/usr/bin/env python3

""" Splits wav files """

__appname__ = 'split_wav.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import numpy as np
import os
from pydub import AudioSegment
import pandas as pd
import subprocess
from subprocess import call
import sys


def split_wav(source, target, length):

    """ Takes a directory of audio files and splits each one into clips of
    specified length (seconds), saving the clips in a specified folder """

    iteration = 1
    clip_length = length * 1000  # pydub works in milliseconds
    files = []

    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith((".WAV", ".wav")):

                basename = (os.path.splitext(file)[0]) # extract hexadecimal timestamp
                print(basename) # so user knows which file is being processed
                uncut_wav = AudioSegment.from_wav(root + '/' + file) # extract audio data
                duration = uncut_wav.duration_seconds # extract clip duration
                no_clips = int(duration // (clip_length / 1000)) # determine number of clips

                i = 0 # clip iterator

                # Iterate through audio data and split
                for i in range(0, no_clips):
                    clip = uncut_wav[(i*clip_length):(clip_length) * (i+1)]
                    new_path = target + '/' + basename + '_' + str(i) + '.wav'
                    clip.export(new_path, format="wav") # save clip
                    i = i + 1

                iteration = iteration + 1 # file iterator

                # Progress report
                progress = int((iteration / total) * 100)
                print(str(progress) + '%')


# User inputs source and target directories at command line
source = str(sys.argv[1])
target = str(sys.argv[2])

# Determine total number of audio files present
total=0
x=[]
for file in os.listdir(str(sys.argv[1])):
    if file.endswith(('.wav', '.WAV')):
        x.append(file)
        total+=1
print('Audio files found: ' +str(total)) # report to user

length = 4

#run split_wav.py /media/jacob/Samsung_external/Indigenous_reserve/Audio3 /media/jacob/Samsung_external/Indigenous_reserve/Cut/Audio3

split_wav(source, target, length)
