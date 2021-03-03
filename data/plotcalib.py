#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 11:14:06 2021

@author: drugo
"""

import pandas as pd
import os
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('auemr_calib.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df.columns = ['Date','fao_cal','fao_not_cal','ps_cal','ps_not_cal','obs']
#df = df.iloc[:,1:]#[['1']]
#df.columns = ['1','2']
print(df.head())


df['diffCal']=df['ps_cal']-df['obs']
df['diffNotCal']=df['ps_not_cal']-df['obs']


#df = df.dropna()
print('RMSE cal:         ',((df.ps_cal - df.obs) ** 2).mean() ** .5)
print('RMSE not cal: ',((df.ps_not_cal - df.obs) ** 2).mean() ** .5)


df[['diffCal']].plot()
df[['diffNotCal']].plot()



dfD = df.resample('1H').agg('mean')
dfDay = dfD[['fao_cal','fao_not_cal','ps_cal','ps_not_cal','obs']].copy()
dfDay = dfDay.resample('1D').agg('sum')

dfDay['month'] = dfDay.index.month

dfDay['year'] = dfDay.index.year

dfDay['SEASON'] = pd.cut(
    (dfDay.index.dayofyear + 11) % 366,
    [0, 91, 183, 275, 366],
    labels=['Winter', 'Spring', 'Summer', 'Fall']
)


#g = sns.pairplot(dfDay[['fao_cal','fao_not_cal','ps_cal','ps_not_cal','obs']],height=2.5)
#g.set(xlim=(0, 8))
#g.set(ylim=(0, 8))
#g.add_legend()

print('RMSE fao cal:        ',((df.fao_cal - df.obs) ** 2).mean() ** .5)
print('RMSE fao not cal:    ',((df.fao_not_cal - df.obs) ** 2).mean() ** .5)
print('RMSE ps cal:         ',((df.ps_cal - df.obs) ** 2).mean() ** .5)
print('RMSE ps not cal:     ',((df.ps_not_cal - df.obs) ** 2).mean() ** .5)

print('#######################')

print('RMSE fao cal:        ',((dfDay.fao_cal - dfDay.obs) ** 2).mean() ** .5)
print('RMSE fao not cal:    ',((dfDay.fao_not_cal - dfDay.obs) ** 2).mean() ** .5)
print('RMSE ps cal:         ',((dfDay.ps_cal - dfDay.obs) ** 2).mean() ** .5)
print('RMSE ps not cal:     ',((dfDay.ps_not_cal - dfDay.obs) ** 2).mean() ** .5)