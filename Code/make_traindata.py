#!/usr/bin/env python3

""" """

__appname__ = 'make_traindata.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

import sys
import pandas as pd
import numpy as np
from shutil import copyfile
import os
import glob

np.random.seed(23)

drive = '/media/jacob/Samsung_external/'

train_dir = drive + sys.argv[1]
num_dirs = len(sys.argv)
dir_list = sys.argv[2:num_dirs]
dir_list = [drive + s for s in dir_list]
colnames = ["ID", "Class", "File"]

def append_jpg(fn):
    return str(fn) + ".jpg"

def append_wav(fn):
    return str(fn) + ".wav"

def append_folder(fn):
    return area + '_' + str(fn)


for dir in dir_list:
    audio_num = os.path.basename(dir)
    area = dir.split("/Samsung_external/", 1)[1]
    area = area.split("/Spectra", 1)[0]


    positives = pd.read_csv(str(dir) + '/confirmed.csv', names=colnames)
    negatives = pd.read_csv(str(dir) + '/false_positives.csv', names=colnames)
    negatives = negatives.sample(n=len(positives), replace=False, random_state=23)
    combined = pd.concat([positives, negatives])
    combined['ID'] = combined['ID'].apply(append_folder)
    #combined[2] = combined[2].apply(append_wav)

    spectra = np.asarray(combined['ID'])

    spectra = str(dir) + '/' + spectra

    i=0
    for filepath in spectra:
        copyfile(spectra[i], train_dir + '/' + os.path.basename(spectra[i]))
        i = i + 1

    combined.to_csv(train_dir + '/' + area + '_' + audio_num + '.csv', index=False, header=None)


path = '/media/jacob/Samsung_external/Train' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.to_csv(train_dir + '/Results/train.csv', index=False, header=None)

    #run make_traindata.py Train SQ258/Spectra/Audio1 SQ258/Spectra/Audio1 SQ283/Spectra/Audio1 SQ282/Spectra/Audio1 La_Balsa/Spectra/Audio1 Rancho_bajo/Spectra/Audio1 Indigenous_reserve/Spectra/Audio1 Gunshot_recordings/Spectra/Audio1
