#!/usr/bin/env python3

"""  """

__appname__ = 'day_of_week.py'
__author__ = 'Jacob Griffiths (jacob.griffiths18@imperial.ac.uk)'
__version__ = '0.0.1'


import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare

SQ258_data = pd.read_csv('/media/jacob/Samsung_external/SQ258/Results/time.csv')
SQ283_data = pd.read_csv('/media/jacob/Samsung_external/SQ283/Results/time.csv')
SQ282_data = pd.read_csv('/media/jacob/Samsung_external/SQ282/Results/time.csv')
La_Balsa_data = pd.read_csv('/media/jacob/Samsung_external/La_Balsa/Results/time.csv')
Rancho_bajo_data = pd.read_csv('/media/jacob/Samsung_external/Rancho_bajo/Results/time.csv')
Indigenous_reserve_data = pd.read_csv('/media/jacob/Samsung_external/Indigenous_reserve/Results/time.csv')

time_data = pd.concat([SQ258_data, SQ283_data, SQ282_data, La_Balsa_data, Rancho_bajo_data, Indigenous_reserve_data])
time_data.to_csv('/media/jacob/Samsung_external/time_data.csv', index=False)


SQ258_confirmed = pd.read_csv('/media/jacob/Samsung_external/SQ258/Results/time_confirmed.csv')
SQ283_confirmed = pd.read_csv('/media/jacob/Samsung_external/SQ283/Results/time_confirmed.csv')
SQ282_confirmed = pd.read_csv('/media/jacob/Samsung_external/SQ282/Results/time_confirmed.csv')
La_Balsa_confirmed = pd.read_csv('/media/jacob/Samsung_external/La_Balsa/Results/time_confirmed.csv')
Rancho_bajo_confirmed = pd.read_csv('/media/jacob/Samsung_external/Rancho_bajo/Results/time_confirmed.csv')
Indigenous_reserve_confirmed = pd.read_csv('/media/jacob/Samsung_external/Indigenous_reserve/Results/time_confirmed.csv')

time_data_confirmed = pd.concat([SQ258_confirmed, SQ283_confirmed, SQ282_confirmed, La_Balsa_confirmed, Rancho_bajo_confirmed, Indigenous_reserve_confirmed])
time_data_confirmed.to_csv('/media/jacob/Samsung_external/time_data_confirmed.csv', index=False)


weekday_data = time_data["Day_of_week"].value_counts()
weekday_data_confirmed = time_data_confirmed["Day_of_week"].value_counts()

divided = (weekday_data_confirmed.divide(weekday_data)) * 100
divided = divided.reindex(index = ['Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

divided.plot.bar(x='Day of week', y='Proportion of gunshots')


# Prepare Data
# df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
# df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.mean())
# df.sort_values('cty', inplace=True)
# df.reset_index(inplace=True)




# Draw plot

fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
ax.vlines(x=divided.index, ymin=0, ymax= divided, color='firebrick', alpha=0.7, linewidth=30)

# Annotate Text
for i, day in enumerate(divided):
    ax.text(i, day+0.5, round(day, 1), horizontalalignment='center')
    ax.set_ylabel('Percentage of gunshots', size = 18)
    ax.set_xlabel('Day of week', size = 18)
    ax.xaxis.set_tick_params(rotation=45)



# Title, Label, Ticks and Ylim
#ax.set_title('Bar Chart for Highway Mileage', fontdict={'size':22})

#plt.xticks(divided.index, rotation=45, horizontalalignment='right', fontsize=14)
fig.tight_layout()


# Add patches to color the X axis labels
# p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
# p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
# fig.add_artist(p1)
# fig.add_artist(p2)
fig.savefig('/media/jacob/Samsung_external/day_of_week.jpg')

chisquare(divided)
