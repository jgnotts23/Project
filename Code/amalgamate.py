#!/usr/bin/env python3

"""  """

__appname__ = 'amalgamate.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'



import pandas as pd
import numpy as np
import os


### Amalgamate confirmed gunshot filepaths
SQ258 = pd.read_csv('/media/jacob/Samsung_external/SQ258/Spectra/Audio1/confirmed.csv', header=None)
SQ283 = pd.read_csv('/media/jacob/Samsung_external/SQ283/Spectra/Audio1/confirmed.csv', header=None)
SQ282 = pd.read_csv('/media/jacob/Samsung_external/SQ282/Spectra/Audio1/confirmed.csv', header=None)
La_Balsa = pd.read_csv('/media/jacob/Samsung_external/La_Balsa/Spectra/Audio1/confirmed.csv', header=None)
Rancho_bajo = pd.read_csv('/media/jacob/Samsung_external/Rancho_bajo/Spectra/Audio1/confirmed.csv', header=None)
Indigenous_reserve = pd.read_csv('/media/jacob/Samsung_external/Indigenous_reserve/Spectra/Audio1/confirmed.csv', header=None)

# SQ258 = np.asarray(SQ258[0])
# SQ283 = np.asarray(SQ283[0])
# SQ282 = np.asarray(SQ282[0])
# La_Balsa = np.asarray(La_Balsa[0])
# Rancho_bajo = np.asarray(Rancho_bajo[0])
# Indigenous_reserve = np.asarray(Indigenous_reserve[0])

SQ258[0] = '/media/jacob/Samsung_external/SQ258/Cut/Audio1/confirmed/' + SQ258[0]
SQ283[0] = '/media/jacob/Samsung_external/SQ283/Cut/Audio1/confirmed/' + SQ283[0]
SQ282[0] = '/media/jacob/Samsung_external/SQ282/Cut/Audio1/confirmed/' + SQ282[0]
La_Balsa[0] = '/media/jacob/Samsung_external/La_Balsa/Cut/Audio1/confirmed/' + La_Balsa[0]
Rancho_bajo[0] = '/media/jacob/Samsung_external/Rancho_bajo/Cut/Audio1/confirmed/' + Rancho_bajo[0]
Indigenous_reserve[0] = '/media/jacob/Samsung_external/Indigenous_reserve/Cut/Audio1/confirmed/' + Indigenous_reserve[0]

SQ258[2] = '/media/jacob/Samsung_external/SQ258/Spectra/Audio1/confirmed/' + SQ258[2]
SQ283[2] = '/media/jacob/Samsung_external/SQ283/Spectra/Audio1/confirmed/' + SQ283[2]
SQ282[2] = '/media/jacob/Samsung_external/SQ282/Spectra/Audio1/confirmed/' + SQ282[2]
La_Balsa[2] = '/media/jacob/Samsung_external/La_Balsa/Spectra/Audio1/confirmed/' + La_Balsa[2]
Rancho_bajo[2] = '/media/jacob/Samsung_external/Rancho_bajo/Spectra/Audio1/confirmed/' + Rancho_bajo[2]
Indigenous_reserve[2] = '/media/jacob/Samsung_external/Indigenous_reserve/Spectra/Audio1/confirmed/' + Indigenous_reserve[2]

#positives = np.concatenate((SQ258, SQ283, SQ282, La_Balsa, Rancho_bajo, Indigenous_reserve))

positives = pd.concat([SQ258, SQ283, SQ282, La_Balsa, Rancho_bajo, Indigenous_reserve])
positives.to_csv('/media/jacob/Samsung_external/Train/Results/positives.csv', index=False)


### Choose blank data
# def get_blank_pathlist (directory):
#     files = os.listdir(directory)
#
#     wavs = []
#     for filename in files:
#         if filename.endswith(".jpg"): # check each of the files for whether or not they end in .wav
#             wavs.append(filename)
#
#
#     wavs = [directory + s for s in wavs]
#
#     wavs = np.asarray(wavs)
#
#     return wavs
#
#
# SQ258_blanks = get_blank_pathlist('/media/jacob/Samsung_external/SQ258/Audio1/Spectra/')
# SQ283_blanks = get_blank_pathlist('/media/jacob/Samsung_external/SQ283/Audio1/Spectra/')
# SQ282_blanks = get_blank_pathlist('/media/jacob/Samsung_external/SQ282/Audio1/Spectra/')
# La_Balsa_blanks = get_blank_pathlist('/media/jacob/Samsung_external/La_Balsa/Audio1/Spectra/')
# Rancho_bajo_blanks = get_blank_pathlist('/media/jacob/Samsung_external/Rancho_bajo/Audio1/Spectra/')
# Indigenous_reserve_blanks = get_blank_pathlist('/media/jacob/Samsung_external/Indigenous_reserve/Audio1/Spectra/')
#
# negatives = np.concatenate((SQ258_blanks, SQ283_blanks, SQ282_blanks, La_Balsa_blanks, Rancho_bajo_blanks, Indigenous_reserve_blanks))
#
# np.random.seed(23)
#
# negatives = np.random.choice(negatives, size = 253, replace = False)
# zeros = np.zeros(len(negatives))
# data=pd.DataFrame({0:zeros,
#                       1:zeros,
#                       2:negatives})
#
# train_data = pd.concat([positives, data])
# new = np.arange(len(train_data))
# train_data[3] = new
# def append_ext(fn):
#     return str(fn) + ".jpg"
#
# train_data[3]=train_data[3].apply(append_ext)
#
# subset = train_data[[2,3]]
#
# subset.to_csv('/media/jacob/Samsung_external/Train/fixed.txt', index=False, sep=' ', header=None)
# train_data.to_csv('/media/jacob/Samsung_external/Train/train.csv', index=False, header=None)
