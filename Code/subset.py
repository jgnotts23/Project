#!/usr/bin/env python3

""" """

__appname__ = 'subset.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


import pandas as pd
import sys
import numpy as np


#data = pd.read_csv(str(sys.argv[1]))

positives = pd.read_csv(str(sys.argv[1]), dtype=str, header=None)
data = pd.read_csv(str(sys.argv[2]), dtype=str)

confirmed = positives.loc[positives[1] == '1']
false_pos = positives.loc[positives[1] == '0']

positives_list = np.asarray(confirmed[0])

def append_ext(fn):
    return str(fn) + ".wav"

data['File']=data['File'].apply(append_ext)

def append_jpg(fn):
    return str(fn) + ".jpg"

data['ID']=data['ID'].apply(append_jpg)

zeros = np.zeros(len(data))
data['Class'] = zeros

data = data[~data['File'].isin(positives_list)]

#confirmed = confirmed[0]

#negatives.to_csv(str(sys.argv[3]) + '/negatives.csv', index=False, header=None)
data.to_csv(str(sys.argv[3]) + '/negatives.csv', index=False, header=None)
confirmed.to_csv(str(sys.argv[3]) + '/confirmed.csv', index=False, header=None)
false_pos.to_csv(str(sys.argv[3]) + '/false_positives.csv', index=False, header=None)


#run subset.py /media/jacob/Samsung_external/SQ258/Spectra/Audio1/positives.csv /media/jacob/Samsung_external/SQ258/Spectra/Audio1/gun_results.csv /media/jacob/Samsung_external/SQ258/Spectra/Audio1
