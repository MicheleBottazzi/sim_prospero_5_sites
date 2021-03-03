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
import goodness_of_fit as gof

df = pd.read_csv('/home/drugo/simProspero/data/AU_Emr/AU_Emr_ElCorr_pos.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=2)
df1 = pd.read_csv('/home/drugo/simProspero/output/AU_Emr/prospero_AU_Emr_calibrato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df2 = pd.read_csv('/home/drugo/simProspero/output/AU_Emr/prospero_AU_Emr_validato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)

df3 = pd.read_csv('/home/drugo/simProspero/output/AU_Emr_latentHeat_FAO_calibrato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df4 = pd.read_csv('/home/drugo/simProspero/output/AU_Emr_latentHeat_FAO_validato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)

df5 = pd.read_csv('/home/drugo/simProspero/output/AU_Emr_latentHeat_PT_calibrato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df6 = pd.read_csv('/home/drugo/simProspero/output/AU_Emr_latentHeat_PT_validato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)

date_index = pd.date_range('2012-01-01 00:00','2013-12-31 23:30', freq='30T')

df = df.iloc[:,1:]
print(df.head())
df[df<0]=0
df1 = df1.iloc[:,1:]
print(df1.head())

df2 = df2.iloc[:,1:]
df3 = df3.iloc[:,1:]
df4 = df4.iloc[:,1:]
df5 = df5.iloc[:,1:]
df6 = df6.iloc[:,1:]

df = df.reindex(date_index)
df1 = df1.reindex(date_index)
df2 = df2.reindex(date_index)
df3 = df3.reindex(date_index)
df4 = df4.reindex(date_index)
df5 = df5.reindex(date_index)
df6 = df6.reindex(date_index)

df = pd.concat([df,df1,df2,df3,df4,df5,df6],axis=1)
df.columns = ['obs','ps_cal','ps_val','fao_cal','fao_val','pt_cal','pt_val']
print(df.head())

print(df.isna().sum())

#df = df.dropna()

dfD = df.resample('1H').agg('mean')
dfDay = dfD.copy()
dfDay = dfDay.resample('1D').agg('sum')

dfDay = dfDay/2.45E6*3600

dfDay['month'] = dfDay.index.month

dfDay['year'] = dfDay.index.year

dfDay['SEASON'] = pd.cut(
    (dfDay.index.dayofyear + 11) % 366,
    [0, 91, 183, 275, 366],
    labels=['Winter', 'Spring', 'Summer', 'Fall']
)

print('#######################')
print('######   HOURLY  ######')
print('#######################')

print('RMSE fao cal:        ',((df.fao_cal - df.obs) ** 2).mean() ** .5)
print('RMSE fao not cal:    ',((df.fao_not_cal - df.obs) ** 2).mean() ** .5)
print('RMSE ps cal:         ',((df.ps_cal - df.obs) ** 2).mean() ** .5)
print('RMSE ps not cal:     ',((df.ps_not_cal - df.obs) ** 2).mean() ** .5)
print('RMSE pt cal:         ',((df.pt_cal - df.obs) ** 2).mean() ** .5)
print('RMSE pt not cal:     ',((df.pt_not_cal - df.obs) ** 2).mean() ** .5)

print('#######################')

print('MAE fao cal:         ',df.eval('fao_cal-obs').abs().mean())
print('MAE fao not cal:     ',df.eval('fao_not_cal-obs').abs().mean())
print('MAE ps cal:          ',df.eval('ps_cal-obs').abs().mean())
print('MAE ps not cal:      ',df.eval('ps_not_cal-obs').abs().mean())
print('MAE pt cal:          ',df.eval('pt_cal-obs').abs().mean())
print('MAE pt not cal:      ',df.eval('pt_not_cal-obs').abs().mean())

print('#######################')

print('R^{2} fao cal:       ',df.fao_cal.corr(df.obs))
print('R^{2} fao not cal:   ',df.fao_not_cal.corr(df.obs))
print('R^{2} ps cal:        ',df.ps_cal.corr(df.obs))
print('R^{2} ps not cal:    ',df.ps_not_cal.corr(df.obs))
print('R^{2} pt cal:        ',df.pt_cal.corr(df.obs))
print('R^{2} pt not cal:    ',df.pt_not_cal.corr(df.obs))
print('#######################')

print('index d fao cal:     ',gof.d(np.array(df.fao_cal),np.array(df.obs)))
print('index d fao not cal: ',gof.d(np.array(df.fao_not_cal),np.array(df.obs)))
print('index d ps cal:      ',gof.d(np.array(df.ps_cal),np.array(df.obs)))
print('index d ps not cal:  ',gof.d(np.array(df.ps_not_cal),np.array(df.obs)))
print('index d pt cal:      ',gof.d(np.array(df.pt_cal),np.array(df.obs)))
print('index d pt not cal:  ',gof.d(np.array(df.pt_not_cal),np.array(df.obs)))


print('#######################')

print('nse fao cal:     ',gof.nse(np.array(df.fao_cal),np.array(df.obs)))
print('nse fao not cal: ',gof.nse(np.array(df.fao_not_cal),np.array(df.obs)))
print('nse ps cal:      ',gof.nse(np.array(df.ps_cal),np.array(df.obs)))
print('nse ps not cal:  ',gof.nse(np.array(df.ps_not_cal),np.array(df.obs)))
print('nse pt cal:      ',gof.nse(np.array(df.pt_cal),np.array(df.obs)))
print('nse pt not cal:  ',gof.nse(np.array(df.pt_not_cal),np.array(df.obs)))

print('#######################')

print('kge fao cal:     ',gof.kge(np.array(df.fao_cal),np.array(df.obs)))
print('kge fao not cal: ',gof.kge(np.array(df.fao_not_cal),np.array(df.obs)))
print('kge ps cal:      ',gof.kge(np.array(df.ps_cal),np.array(df.obs)))
print('kge ps not cal:  ',gof.kge(np.array(df.ps_not_cal),np.array(df.obs)))
print('kge pt cal:      ',gof.kge(np.array(df.pt_cal),np.array(df.obs)))
print('kge pt not cal:  ',gof.kge(np.array(df.pt_not_cal),np.array(df.obs)))
print('#######################')
print('######   DAILY   ######')
print('#######################')

print('RMSE fao cal:        ',((dfDay.fao_cal - dfDay.obs) ** 2).mean() ** .5)
print('RMSE fao not cal:    ',((dfDay.fao_val - dfDay.obs) ** 2).mean() ** .5)
print('RMSE ps cal:         ',((dfDay.ps_cal - dfDay.obs) ** 2).mean() ** .5)
print('RMSE ps not cal:     ',((dfDay.ps_val - dfDay.obs) ** 2).mean() ** .5)
print('RMSE pt cal:         ',((dfDay.pt_cal - dfDay.obs) ** 2).mean() ** .5)
print('RMSE pt not cal:     ',((dfDay.pt_val - dfDay.obs) ** 2).mean() ** .5)

print('#######################')

print('MAE fao cal:         ',dfDay.eval('fao_cal-obs').abs().mean())
print('MAE fao not cal:     ',dfDay.eval('fao_val-obs').abs().mean())
print('MAE ps cal:          ',dfDay.eval('ps_cal-obs').abs().mean())
print('MAE ps not cal:      ',dfDay.eval('ps_val-obs').abs().mean())
print('MAE pt cal:          ',dfDay.eval('pt_cal-obs').abs().mean())
print('MAE pt not cal:      ',dfDay.eval('pt_val-obs').abs().mean())

print('#######################')

print('R^{2} fao cal:       ',dfDay.fao_cal.corr(dfDay.obs))
print('R^{2} fao not cal:   ',dfDay.fao_val.corr(dfDay.obs))
print('R^{2} ps cal:        ',dfDay.ps_cal.corr(dfDay.obs))
print('R^{2} ps not cal:    ',dfDay.ps_val.corr(dfDay.obs))
print('R^{2} pt cal:        ',dfDay.pt_cal.corr(dfDay.obs))
print('R^{2} pt not cal:    ',dfDay.pt_val.corr(dfDay.obs))

print('#######################')

print('index d fao cal:     ',gof.d(np.array(dfDay.fao_cal),np.array(dfDay.obs)))
print('index d fao not cal: ',gof.d(np.array(dfDay.fao_val),np.array(dfDay.obs)))
print('index d ps cal:      ',gof.d(np.array(dfDay.ps_cal),np.array(dfDay.obs)))
print('index d ps not cal:  ',gof.d(np.array(dfDay.ps_val),np.array(dfDay.obs)))
print('index d pt cal:      ',gof.d(np.array(dfDay.pt_cal),np.array(dfDay.obs)))
print('index d pt not cal:  ',gof.d(np.array(dfDay.pt_val),np.array(dfDay.obs)))

print('#######################')

print('nse fao cal:     ',gof.nse(np.array(dfDay.fao_cal),np.array(dfDay.obs)))
print('nse fao not cal: ',gof.nse(np.array(dfDay.fao_val),np.array(dfDay.obs)))
print('nse ps cal:      ',gof.nse(np.array(dfDay.ps_cal),np.array(dfDay.obs)))
print('nse ps not cal:  ',gof.nse(np.array(dfDay.ps_val),np.array(dfDay.obs)))
print('nse pt cal:      ',gof.nse(np.array(dfDay.pt_cal),np.array(dfDay.obs)))
print('nse pt not cal:  ',gof.nse(np.array(dfDay.pt_val),np.array(dfDay.obs)))

print('#######################')

print('kge fao cal:     ',gof.kge(np.array(dfDay.fao_cal),np.array(dfDay.obs)))
print('kge fao not cal: ',gof.kge(np.array(dfDay.fao_val),np.array(dfDay.obs)))
print('kge ps cal:      ',gof.kge(np.array(dfDay.ps_cal),np.array(dfDay.obs)))
print('kge ps not cal:  ',gof.kge(np.array(dfDay.ps_val),np.array(dfDay.obs)))
print('kge pt cal:      ',gof.kge(np.array(dfDay.pt_cal),np.array(dfDay.obs)))
print('kge pt not cal:  ',gof.kge(np.array(dfDay.pt_val),np.array(dfDay.obs)))

print('#######################')

dfMonth = dfDay[['obs','ps_cal','ps_val','fao_cal','fao_val','pt_cal','pt_val']].copy()
dfYear = dfDay[['obs','ps_cal','ps_val','fao_cal','fao_val','pt_cal','pt_val']].copy()

dfMonth = dfMonth.resample('1MS').agg('sum')
dfYear = dfYear.resample('1YS').agg('sum')

dfMonth['year'] = dfMonth.index.year
dfYear['year'] = dfYear.index.year
dfPrpY = pd.read_csv('/home/drugo/simProspero/data/AU_Emr/AU_Emr_Prp_Year.csv', index_col='TIMESTAMP')

#dfPrpY = dfPrpY.iloc[2:,:]
dfPrpY.columns = ['prp']
dfYear = pd.concat([dfYear,dfPrpY], axis=1)


plt.style.use('default')
fig1 = plt.figure(figsize=(35,10))
ax1 = fig1.add_subplot(111)

dfMonth.plot(y=['obs','ps_cal','ps_val','fao_cal','fao_val',
                'pt_cal','pt_val'],kind='line', ax = ax1, 
             colormap='tab10',linewidth=5, rot = -0,stacked=False,
             label = ['Observed','PS calibrated','PS uncalibrated','FAO calibrated',
                      'FAO uncalibrated','PT calibrated','PT uncalibrated'])


ax1.set_ylabel('ET [mm]',fontsize=30,fontweight='bold')
ax1.set_xlabel('Year',fontsize=30,fontweight='bold')
ax1.legend(loc='upper center',fontsize=20, bbox_to_anchor=(0.5, 1.075),ncol=7,
           handleheight=1, labelspacing=0.05,frameon=False)      

# axy.legend(loc='upper center',fontsize=30, bbox_to_anchor=(0.85, 1.15),ncol=2,handleheight=1, labelspacing=0.05)  
plt.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=30)
#fig1.savefig('plot_AU_Emr.png')


plt.style.use('default')
fig2 = plt.figure(figsize=(35,10))
ax2 = fig2.add_subplot(111)
fig2.suptitle('US Vaira Ranch',fontsize=30)

dfYear.plot(y=['obs','ps_cal','ps_val','fao_cal','fao_val',
               'pt_cal','pt_val','prp'],x='year',kind='bar', ax = ax2, 
             colormap='tab10',edgecolor=".3",linewidth=1.5, rot = -0,stacked=False,
             label = ['Observed','PS calibrated','PS uncalibrated','FAO calibrated',
                      'FAO uncalibrated','PT calibrated','PT uncalibrated','Precipitation'])


ax2.set_ylabel('ET [mm]',fontsize=30,fontweight='bold')
ax2.set_xlabel('Year',fontsize=30,fontweight='bold')
ax2.legend(loc='upper center',fontsize=20, bbox_to_anchor=(0.5, 1.075),ncol=8,
           handleheight=1, labelspacing=0.05,frameon=False)   

# axy.legend(loc='upper center',fontsize=30, bbox_to_anchor=(0.85, 1.15),ncol=2,handleheight=1, labelspacing=0.05)  
plt.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=30)
#fig2.savefig('histogram_AU_Emr.png')




fig, axes = plt.subplots(1, 7, figsize=(35, 5), sharey=True)
fig.suptitle('US Vaira Ranch',fontsize=20)

sns.kdeplot(data=dfDay, ax=axes[0],x=dfDay.obs, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[1],x=dfDay.ps_cal, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[2],x=dfDay.ps_val, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[3],x=dfDay.fao_cal, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[4],x=dfDay.fao_val, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[5],x=dfDay.pt_cal, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[6],x=dfDay.pt_val, hue="SEASON")
#axes[0].set_title('Observed')
axes[0].set_xlabel('Observed',fontsize=15)
axes[1].set_xlabel('PS calibrated',fontsize=15)
axes[2].set_xlabel('PS uncalibrated',fontsize=15)
axes[3].set_xlabel('FAO calibrated',fontsize=15)
axes[4].set_xlabel('FAO uncalibrated',fontsize=15)
axes[5].set_xlabel('PT calibrated',fontsize=15)
axes[6].set_xlabel('PT uncalibrated',fontsize=15)


for i in range(0,7):
    #axes[i].legend().set_visible(False)
    axes[i].set_ylim((0,0.4))
    axes[i].set_xlim((-1,8))
    axes[i].set_ylabel('Density',fontsize=15)
    axes[i].tick_params(axis='both', which='major', labelsize=15)

#fig.savefig('distribution_AU_Emr.png')

