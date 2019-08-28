#!/usr/bin/env python3

""" """

__appname__ = 'compare_tests.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


import pandas as pd
import sys
import numpy as np


positives = pd.read_csv('/media/jacob/Samsung_external/Indigenous_reserve/Spectra/positives.csv', dtype=str, header=None)
positives2 = pd.read_csv('/media/jacob/Samsung_external/Indigenous_reserve/Spectra/updated_positives2.csv', dtype=str, header=None)

pos_images = np.asarray(positives[2])
pos2_images = np.asarray(positives2[0])

same = np.intersect1d(pos_images, pos2_images)

print(len(pos_images))
print(len(pos2_images))
print(len(same))
