#!/usr/bin/env python3

""" """

__appname__ = 'split_wav.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

import glob
import sys
import os
from os import walk
from pydub import AudioSegment
import numpy as np

mypath = str(sys.argv[1]) #+ '/*.WAV'

f = np.asarray(os.listdir(mypath))

f_split = np.array_split(f, 100)

def combine_wav(files, iter):

    combined_sound = AudioSegment.empty()

    for filename in files:
        combined_sound += AudioSegment.from_wav(mypath + '/' + filename)

        outfile = mypath + "/combined" + str(iter) + ".wav"
        combined_sound.export(outfile, format='wav')

iter = 0
for f in f_split:
    combine_wav(f, iter)
    iter = iter + 1
    progress = str(iter * 1) + '%'
    print(progress)
