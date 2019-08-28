#!/usr/bin/env python3

""" """

__appname__ = 'time_extract.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'

from    datetime import datetime # used by filename_to_localdatetime
from    pytz import timezone # used by filename_to_localdatetime
import  pytz
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def filename_to_localdatetime(filename):
    """
    Extracts datetime of recording in Costa Rica time from hexadecimal file name.
    Example call: filename_to_localdatetime('5A3AD5B6')
    """
    time_stamp = int(filename, 16)
    naive_utc_dt = datetime.fromtimestamp(time_stamp)
    aware_utc_dt = naive_utc_dt.replace(tzinfo=pytz.UTC)
    cst = timezone('America/Costa_Rica')
    cst_dt = aware_utc_dt.astimezone(cst)
    return cst_dt

#filename_to_localdatetime('5A3AD5B6')



def get_time(filepath):
    filepath = filepath.split("Samsung_external/", 1)[1]
    basename = os.path.basename(filepath)
    dirname = os.path.dirname(filepath)

    file = basename[0:8]
    area = filepath.split("/Cut/", 1)[0]
    #print(area)

    reg_time = filename_to_localdatetime(file)

    return area, reg_time, file


#area, reg_time = get_time(wav_files[3])

def time_data(dataframe):

    colnames = ('Area', 'File', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Timestamp', 'Day_of_week')
    data = pd.DataFrame(columns=(colnames))
    i = 0

    for filepath in dataframe:
        filename = filepath.split("/Audio1/", 1)[1]
        Area, reg_time, File = get_time(filepath)
        new_row = pd.Series([Area, filename, reg_time.year, reg_time.month, reg_time.day, reg_time.hour, reg_time.minute, reg_time.second, reg_time.timestamp(), weekDays[reg_time.weekday()]], index=(colnames))
        data.loc[i] = new_row
        i = i + 1

    return data



#positives = pd.read_csv('/media/jacob/Samsung_external/Train/positives.csv', header=None)
#wav_files = positives[0]
#del wav_files[0]


# def append_jpg(fn):
#     return str(fn).replace('.jpg.jpg', '.jpg')
# def append_wav(fn):
#     return str(fn).replace('.wav.wav', '.wav')
#
# data[0]=data[0].apply(append_jpg)
# data[2]=data[2].apply(append_wav)
#
# data.to_csv(str(sys.argv[1]), header=None, index=False)


spec_dir = '/media/jacob/Samsung_external/' + str(sys.argv[1]) + '/Spectra/Audio1/'
wav_dir = '/media/jacob/Samsung_external/' + str(sys.argv[1]) + '/Cut/Audio1/'
results_dir = '/media/jacob/Samsung_external/' + str(sys.argv[1]) + '/Results/time_confirmed.csv'

colnames = ("ID", "Class", "File")

data = pd.read_csv(spec_dir + 'confirmed.csv', header=0, dtype=str, names=colnames)
wav_files = data['File']
wav_files = [wav_dir + s for s in wav_files]



weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
data = time_data(wav_files)
data.to_csv(results_dir , index=False)

#run time_extract.py /media/jacob/Samsung_external/SQ258/Spectra/Audio1/gun_results.csv /media/jacob/Samsung_external/SQ258/Results

### Time analysis ###
# Hour of Day
# hours = data['Hour']
# unique, counts = np.unique(hours, return_counts=True)
#
# xlabels = np.arange(24)
# y_pos = np.arange(len(xlabels))
#
# fig = plt.figure()
# plt.bar(unique, counts)
# plt.xticks(y_pos, xlabels)
# plt.xlabel('Hour of day')
# plt.ylabel('No. of gunshots')
# fig.savefig('/media/jacob/Samsung_external/Train/hour_of_day.jpg')
#
# # Day of week
# Day_of_week = data['Day_of_week']
# unique, counts = np.unique(Day_of_week, return_counts=True)
#
# xlabels = weekDays
# y_pos = np.arange(len(xlabels))
#
# fig = plt.figure()
# plt.bar(unique, counts)
# plt.xticks(y_pos, xlabels)
# plt.xlabel('Day_of_week')
# plt.ylabel('No. of gunshots')
# fig.savefig('/media/jacob/Samsung_external/Train/Day_of_week.jpg')
