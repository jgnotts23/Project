#!/usr/bin/env python3

""" Project first attempt """

__appname__ = 'first_try.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import IPython.display as ipd
import librosa
import librosa.display
import os
import pandas as pd
import glob 
import matplotlib.pyplot as plt

ipd.Audio('../Data/Cut_Gunshot_Recordings1/002/58FBD3C1.wav')

# Make numpy array of audio file and corresponding sampling rate
data, sampling_rate = librosa.load('../Data/Cut_Gunshot_Recordings1/002/58FBC581.wav')


### Represent as a waveform ###
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()



from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('../Data/Cut_Gunshot_Recordings1/002/58FBC581.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()



data = wave.open('../Data/Cut_Gunshot_Recordings1/002/58FBC581.wav')

import os
import wave

import pylab
def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('spectrogram.png')
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate