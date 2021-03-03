#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os


os.chdir('/home/drugo/simProspero/data/')
site_name = 'GL_Zah'
data_start = '2000-02-18'
data_end = '2014-11-25 23:30'

os.chdir(site_name)


df = pd.read_csv(site_name+'_LAI_DD.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=0)
df = df[['MOD15A2H_006_Lai_500m']]

# df2 = df[['MCD15A2H_006_Lai_500m']].astype(float)




date_index2 = pd.date_range(data_start,data_end, freq='30T')

df.index = pd.to_datetime(df.index)

df4 = df.resample('30T').interpolate('linear')
df4.plot()

df4.to_csv(site_name+'_LAI.csv',na_rep=-9999)



