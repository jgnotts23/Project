#!/usr/bin/env python3

"""  """

__appname__ = 'comparitive_tables.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

import pandas as pd


def getdata (folder):
    filepath = '/media/jacob/Samsung_external/' + folder + '/Spectra/Audio1'
    Belize_data = pd.read_csv(filepath + '/positives.csv', names=colnames)
    Belize_Jenna_data = pd.read_csv(filepath + '/Combined_model.csv', names=colnames)
    Jenna_data = pd.read_csv(filepath + '/Jenna_model.csv', names=colnames)
    Falsepos_data = pd.read_csv(filepath + '/false_positives.csv', names=colnames)

    return Belize_data, Belize_Jenna_data, Jenna_data, Falsepos_data

colnames = ("ID", "Class", "File")
SQ258_Belize, SQ258_Belize_Jenna, SQ258_Jenna, SQ258_Falsepos = getdata('SQ258')
SQ283_Belize, SQ283_Belize_Jenna, SQ283_Jenna, SQ283_Falsepos = getdata('SQ283')
SQ282_Belize, SQ282_Belize_Jenna, SQ282_Jenna, SQ282_Falsepos = getdata('SQ282')
La_Balsa_Belize, La_Balsa_Belize_Jenna, La_Balsa_Jenna,La_Balsa_Falsepos = getdata('La_Balsa')
Rancho_bajo_Belize, Rancho_bajo_Belize_Jenna, Rancho_bajo_Jenna, Rancho_bajo_Falsepos = getdata('Rancho_bajo')
Indigenous_reserve_Belize, Indigenous_reserve_Belize_Jenna, Indigenous_reserve_Jenna, Indigenous_reserve_Falsepos = getdata('Indigenous_reserve')

columns = ("Belize", "Osa_peninsula", "Combined", "Osa_false")
index = ("SQ258", "SQ283", "SQ282", "La_Balsa", "Rancho_bajo", "Indigenous_reserve")
df = pd.DataFrame(index=index, columns=columns)
df.loc["SQ258"] = ((len(SQ258_Belize)), (len(SQ258_Jenna)), (len(SQ258_Belize_Jenna)), (len(SQ258_Falsepos)))
df.loc["SQ283"] = ((len(SQ283_Belize)), (len(SQ283_Jenna)), (len(SQ283_Belize_Jenna)), (len(SQ283_Falsepos)))
df.loc["SQ282"] = ((len(SQ282_Belize)), (len(SQ282_Jenna)), (len(SQ282_Belize_Jenna)), (len(SQ282_Falsepos)))
df.loc["La_Balsa"] = ((len(La_Balsa_Belize)), (len(La_Balsa_Jenna)), (len(La_Balsa_Belize_Jenna)), (len(La_Balsa_Falsepos)))
df.loc["Rancho_bajo"] = ((len(Rancho_bajo_Belize)), (len(Rancho_bajo_Jenna)), (len(Rancho_bajo_Belize_Jenna)), (len(Rancho_bajo_Falsepos)))
df.loc["Indigenous_reserve"] = ((len(Indigenous_reserve_Belize)), (len(Indigenous_reserve_Jenna)), (len(Indigenous_reserve_Belize_Jenna)), (len(Indigenous_reserve_Falsepos)))

df.to_csv('~/Documents/Project/Results/model_comparison.csv')
