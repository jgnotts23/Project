#!/usr/bin/env python3

""" Merging gunshots with Costa Rica background """

__appname__ = 'audio_merge.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

from pydub import AudioSegment

gunshot = AudioSegment.from_file('../Data/Gunshot_recordings/Cut_Gunshot_Recordings/Cut_Gunshot_Recordings1/002/58FBD3C1.wav')
blank = AudioSegment.from_file('../Data/5C37B069_0.wav')


clip_length = 4000
duration = blank.duration_seconds
no_clips = int(duration // (4000 / 1000))

i = 0
for i in range(0, no_clips):
    clip = blank[(i*clip_length):(clip_length) * (i+1)]
    new_path = '../Data/5C37B069' + '_' + str(i) + '.wav'
    clip.export(new_path, format="wav")
    i = i + 1


combined = gunshot.overlay(blank)
combined.export('../Data/combined.wav', format='wav')


import urllib.request
link = 'https://imperialcollegelondon.box.com/s/whsy7zk0seebzx6zgikj3kosvjgpbn5z'
f = urllib.request.urlretrieve(link, '~/Desktop')
myfile = f.read()
print(myfile)
