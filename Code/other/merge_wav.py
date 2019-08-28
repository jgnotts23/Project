#!/usr/bin/env python3

""" """

__appname__ = 'merge_wav.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

from pydub import AudioSegment
import os
import sys
from os import walk

def merge_wav(gunshot_filepath, blank_filepath, new_filepath):
    gunshot = AudioSegment.from_file(gunshot_filepath)
    blank = AudioSegment.from_file(blank_filepath)

    combined = blank.overlay(gunshot)

    combined.export(new_filepath, format='wav')


gunshot_dir = str(sys.argv[1])
blank_dir = str(sys.argv[2])
output_dir = str(sys.argv[3])

blank_data=[]
for root, dirs, files in os.walk(blank_dir):
    for file in files:
        if file.endswith((".WAV", ".wav")):
            blank_data.append(root + '/' + file)

i = 0
for root, dirs, files in os.walk(gunshot_dir):
    for file in files:
        if file.endswith((".WAV", ".wav")):

            gunshot_filepath = root + '/' + file
            blank_filepath = blank_data[i]
            output_filepath = output_dir
            gun_basename = (os.path.splitext(file)[0])
            blank_basename = os.path.splitext(os.path.basename(blank_filepath))[0]
            new_name = '/' + gun_basename + '_' + blank_basename
            new_filepath = output_dir + new_name + '.wav'

            merge_wav(gunshot_filepath, blank_filepath, new_filepath)

            i = i + 1

#run merge_wav.py /media/jacob/Samsung_external/Test/Cut_Gunshot_Recordings /media/jacob/Samsung_external/Test/Cut /media/jacob/Samsung_external/Test/Merged/
