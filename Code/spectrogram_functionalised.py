#!/usr/bin/env python3

""" Making spectrogram.py more general """

__appname__ = 'spectrogram_functionalised.py'
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
import random
import subprocess
from subprocess import call


tps://www.nba.com
tps://www.nba.com
tps://www.nba.com
tps://www.nba.com
Samsung = "/media/jgnotts23/Samsung_external/"

random.seed(23)

### Cut audio into specified chunks ###
def split_wav(source, target, length, sample_size):

    """ Splits a directory of audio file into chunks of specified length (seconds) 
    and saves the clips in a specified folder """

    cmd = 'ls -1 ' + source + ' | wc -l > count.txt'
    number_files = subprocess.call(cmd, shell=True)

    number_files = int(open("count.txt").readline().rstrip())
    sample_size = number_files

    sample = random.sample(range(number_files + 1), sample_size)
    iteration = 1
    clip_length = length * 1000  # pydub works in milliseconds
    files = []

    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith((".WAV", ".wav")):
                print(file + ' done!')
                if iteration in sample:
                    if os.path.getsize(root + '/' + file):

                        basename = (os.path.splitext(file)[0])
                        uncut_wav = AudioSegment.from_wav(root + '/' + file)
                        duration = uncut_wav.duration_seconds
                        no_clips = int(duration // (clip_length / 1000))

                        i = 0
                        for i in range(0, no_clips):
                            clip = uncut_wav[(i*clip_length):(clip_length) * (i+1)]
                            new_path = target + basename + '_' + str(i) + '.wav'
                            clip.export(new_path, format="wav")
                            i = i + 1
                        
                        iteration = iteration + 1

                    else:
                        continue
        
                else:
                    iteration = iteration + 1
                    continue






test = pd.DataFrame(columns=('ID', 'Class', 'File'))
#train = pd.DataFrame(columns=('ID', 'Class', 'File'))


#### Make spectrograms ###
def make_spectra(source, target):

    files = []
    i = 1
    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith((".WAV", ".wav")):
                name = (os.path.splitext(file)[0])
                plot.interactive(False)
                clip, sample_rate = librosa.load(root + '/' + file, sr=None)
                fig = plot.figure(figsize=[0.72, 0.72])
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
                librosa.display.specshow(
                    librosa.power_to_db(S, ref=np.max))

                filename = target + str(i) + '.jpg'
                test.at[i, 'ID'] = i
                test.at[i, 'Class'] = 1
                test.at[i, 'File'] = name

                plot.savefig(filename, dpi=400,
                                bbox_inches='tight', pad_inches=0)
                plot.close()
                fig.clf()
                plot.close(fig)
                plot.close('all')
                i = i + 1
                print(name + ' done!')
                del filename, name, clip, sample_rate, fig, ax, S


test = pd.DataFrame(columns=('ID', 'Class', 'File'))
split_wav("/media/jgnotts23/Samsung_external/SQ258/Audio1", "/media/jgnotts23/Samsung_external/SQ258/Audio1_cut/", 4, 5000)
make_spectra("/media/jgnotts23/Samsung_external/SQ258/Audio1_cut", "/media/jgnotts23/Samsung_external/Results/Audio1/")
test.to_csv('/media/jgnotts23/Samsung_external/Results/Audio1/Audio1.csv', index=False)

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
split_wav("/media/jgnotts23/Samsung_external/SQ258/Audio2", "/media/jgnotts23/Samsung_external/SQ258/Audio2_cut/", 4, 5000)
make_spectra("/media/jgnotts23/Samsung_external/SQ258/Audio2_cut", "/media/jgnotts23/Samsung_external/Results/Audio2/")
test.to_csv('/media/jgnotts23/Samsung_external/Results/Audio2/Audio2.csv', index=False)

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
split_wav("/media/jgnotts23/Samsung_external/SQ258/Audio3", "/media/jgnotts23/Samsung_external/SQ258/Audio3_cut/", 4, 5000)
make_spectra("/media/jgnotts23/Samsung_external/SQ258/Audio3_cut", "/media/jgnotts23/Samsung_external/Results/Audio3/")
test.to_csv('/media/jgnotts23/Samsung_external/Results/Audio3/Audio3.csv', index=False)

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
split_wav("/media/jgnotts23/Samsung_external/SQ258/Audio4", "/media/jgnotts23/Samsung_external/SQ258/Audio4_cut/", 4, 5000)
make_spectra("/media/jgnotts23/Samsung_external/SQ258/Audio4_cut", "/media/jgnotts23/Samsung_external/Results/Audio4/")
test.to_csv('/media/jgnotts23/Samsung_external/Results/Audio4/Audio4.csv', index=False)

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
split_wav("/media/jgnotts23/Samsung_external/SQ258/Audio5", "/media/jgnotts23/Samsung_external/SQ258/Audio5_cut/", 4, 5000)
make_spectra("/media/jgnotts23/Samsung_external/SQ258/Audio5_cut", "/media/jgnotts23/Samsung_external/Results/Audio5/")
test.to_csv('/media/jgnotts23/Samsung_external/Results/Audio5/Audio5.csv', index=False)

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
split_wav("/media/jgnotts23/Samsung_external/SQ258/Audio6", "/media/jgnotts23/Samsung_external/SQ258/Audio6_cut/", 4, 5000)
make_spectra("/media/jgnotts23/Samsung_external/SQ258/Audio6_cut", "/media/jgnotts23/Samsung_external/Results/Audio6/")
test.to_csv('/media/jgnotts23/Samsung_external/Results/Audio6/Audio6.csv', index=False)

test = pd.DataFrame(columns=('ID', 'Class', 'File'))
split_wav("/media/jgnotts23/Samsung_external/SQ258/Audio7", "/media/jgnotts23/Samsung_external/SQ258/Audio7_cut/", 4, 5000)
make_spectra("/media/jgnotts23/Samsung_external/SQ258/Audio7_cut", "/media/jgnotts23/Samsung_external/Results/Audio7/")
test.to_csv('/media/jgnotts23/Samsung_external/Results/Audio7/Audio7.csv', index=False)

