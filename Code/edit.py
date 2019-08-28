#!/usr/bin/env python3

"""  """

__appname__ = 'edit.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

import pandas as pd
import sys

data = pd.read_csv(str(sys.argv[1]),dtype=str)
#testdf = pd.concat([positives, negatives])
#test_filenames = np.asarray(testdf['File'])

data['ID'] = [s + '.jpg' for s in data['ID']]
data['File'] = [s + '.wav' for s in data['File']]

data.to_csv(str(sys.argv[1]))

#run edit.py '/media/jacob/Samsung_external/SQ282/Spectra/Audio1/gun_results.csv'
