#!/usr/bin/env python3

"""  """

__appname__ = 'rename.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

import pandas as pd
import numpy as np
import os

data = pd.read_csv('/media/jacob/Samsung_external/SQ258/Spectra/gun_results.csv')

def append_wav(fn):
    return str(fn) + ".wav"

data['File']=data['File'].apply(append_wav)

def append_jpg(fn):
    return str(fn) + ".jpg"

data['ID']=data['ID'].apply(append_jpg)

folder = 'SQ258_'
