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

#def_site = "AU_Dry"
def_site = "US_Cop"
#def_site = "US_Var"
#def_site = "IT_Tor"
#def_site = "GL_Zah"
file=open(def_site+'_performance.txt','w')
file.close()
file=open(def_site+'_performance.txt','a')

#index = pd.date_range('2008-01-01 00:00', '2014-12-31 23:30', freq='30T')
index = pd.date_range('2001-01-01 00:00', '2007-12-31 23:00', freq='60T')
#index = pd.date_range('2000-01-01 00:00', '2014-12-31 23:30', freq='30T')
#index = pd.date_range('2008-01-01 00:00', '2014-12-31 23:30', freq='30T')
#index = pd.date_range('2000-01-01 00:00', '2014-12-31 23:30', freq='30T')
#df = pd.read_csv('/home/drugo/simProspero/data/'+def_site+'/'+def_site+'_El.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=2)
df = pd.read_csv('/home/drugo/simProspero/data/'+def_site+'/'+def_site+'_ElCorr.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=2)

df1 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/prospero_'+def_site+'_validato2.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df2 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/prospero_'+def_site+'_calibrato2.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df3 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/prospero_'+def_site+'.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)

df4 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/'+def_site+'_latentHeat_FAO_validato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df5 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/'+def_site+'_latentHeat_FAO_calibrato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df6 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/'+def_site+'_latentHeat_FAO.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)

df7 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/'+def_site+'_latentHeat_PT_validato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df8 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/'+def_site+'_latentHeat_PT_calibrato.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)
df9 = pd.read_csv('/home/drugo/simProspero/output/'+def_site+'/'+def_site+'_latentHeat_PT.csv', na_values=-9999, parse_dates=[1], index_col=[1], skiprows=6)


df = df.iloc[:,1:]
df.loc['2004-01-01 00:00':'2005-12-31 23:00']=np.nan
print(df.head())
df[df<0]=np.nan
df1 = df1.iloc[:,1:].reindex(index)
print(df1.head())

df2 = df2.iloc[:,1:].reindex(index)
df3 = df3.iloc[:,1:].reindex(index)
df4 = df4.iloc[:,1:].reindex(index)
df5 = df5.iloc[:,1:].reindex(index)
df6 = df6.iloc[:,1:].reindex(index)
df7 = df7.iloc[:,1:].reindex(index)
df8 = df8.iloc[:,1:].reindex(index)
df9 = df9.iloc[:,1:].reindex(index)

df = pd.concat([df,df1,df2,df3,df4,df5,df6,df7,df8,df9],axis=1)
df.columns = ['obs','ps_val','ps_cal','ps','fao_val','fao_cal','fao','pt_val','pt_cal','pt']
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

'''file.write('#############\n')
    file.write('Precipitation:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_prp.shape)+'\n')'''
file.write('#######################\n')
file.write('######   HOURLY  ######\n')
file.write('#######################\n')

file.write('RMSE ps validated:          '+str(((df.ps_val - df.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE ps calibrated:         '+str(((df.ps_cal - df.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE ps:                    '+str(((df.ps - df.obs) ** 2).mean() ** .5)+'\n')
file.write('\n')


file.write('RMSE fao validated:         '+str(((df.fao_val - df.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE fao calibrated:        '+str(((df.fao_cal - df.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE fao:                   '+str(((df.fao - df.obs) ** 2).mean() ** .5)+'\n')
file.write('\n')

file.write('RMSE pt validated:          '+str(((df.pt_val - df.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE pt calibrated:         '+str(((df.pt_cal - df.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE pt:                    '+str(((df.pt - df.obs) ** 2).mean() ** .5)+'\n')
file.write('\n')

file.write('#######################\n \n')

file.write('MAE ps validated:           '+str(df.eval('ps_val-obs').abs().mean())+'\n')
file.write('MAE ps calibrated:          '+str(df.eval('ps_cal-obs').abs().mean())+'\n')
file.write('MAE ps:                     '+str(df.eval('ps-obs').abs().mean())+'\n')
file.write('\n')

file.write('MAE fao validated:          '+str(df.eval('fao_val-obs').abs().mean())+'\n')
file.write('MAE fao calibrated:         '+str(df.eval('fao_cal-obs').abs().mean())+'\n')
file.write('MAE fao :                   '+str(df.eval('fao-obs').abs().mean())+'\n')
file.write('\n')

file.write('MAE pt validated:           '+str(df.eval('pt_val-obs').abs().mean())+'\n')
file.write('MAE pt calibrated:          '+str(df.eval('pt_cal-obs').abs().mean())+'\n')
file.write('MAE pt:                     '+str(df.eval('pt-obs').abs().mean())+'\n')
file.write('\n')

file.write('#######################\n \n')

file.write('R^{2} ps validated:         '+str(df.ps_val.corr(df.obs))+'\n')
file.write('R^{2} ps calibrated:        '+str(df.ps_cal.corr(df.obs))+'\n')
file.write('R^{2} ps:                   '+str(df.ps.corr(df.obs))+'\n')
file.write('\n')

file.write('R^{2} fao validated:        '+str(df.fao_val.corr(df.obs))+'\n')
file.write('R^{2} fao calibrated:       '+str(df.fao_cal.corr(df.obs))+'\n')
file.write('R^{2} fao:                  '+str(df.fao.corr(df.obs))+'\n')
file.write('\n')

file.write('R^{2} pt validated:         '+str(df.pt_val.corr(df.obs))+'\n')
file.write('R^{2} pt calibrated:        '+str(df.pt_cal.corr(df.obs))+'\n')
file.write('R^{2} pt:                   '+str(df.pt.corr(df.obs))+'\n')
file.write('\n')

file.write('#######################\n \n')

file.write('index d ps validated:       '+str(gof.d(np.array(df.ps_val),np.array(df.obs)))+'\n')
file.write('index d ps calibrated:      '+str(gof.d(np.array(df.ps_cal),np.array(df.obs)))+'\n')
file.write('index d ps:                 '+str(gof.d(np.array(df.ps),np.array(df.obs)))+'\n')
file.write('\n')

file.write('index d fao validated:      '+str(gof.d(np.array(df.fao_val),np.array(df.obs)))+'\n')
file.write('index d fao calibrated:     '+str(gof.d(np.array(df.fao_cal),np.array(df.obs)))+'\n')
file.write('index d fao:                '+str(gof.d(np.array(df.fao),np.array(df.obs)))+'\n')
file.write('\n')

file.write('index d pt validated:       '+str(gof.d(np.array(df.pt_val),np.array(df.obs)))+'\n')
file.write('index d pt calibrated:      '+str(gof.d(np.array(df.pt_cal),np.array(df.obs)))+'\n')
file.write('index d pt:                 '+str(gof.d(np.array(df.pt),np.array(df.obs)))+'\n')
file.write('\n')


file.write('#######################\n \n')

'''file.write('nse ps validated:       '+str(gof.nse(np.array(df.ps_val),np.array(df.obs))))
file.write('nse ps calibrated:      '+str(gof.nse(np.array(df.ps_cal),np.array(df.obs))))
file.write('nse ps:                 '+str(gof.nse(np.array(df.ps),np.array(df.obs))))

file.write('nse fao validated:      '+str(gof.nse(np.array(df.fao_val),np.array(df.obs))))
file.write('nse fao calibrated:     '+str(gof.nse(np.array(df.fao_cal),np.array(df.obs))))
file.write('nse fao:                '+str(gof.nse(np.array(df.fao),np.array(df.obs))))

file.write('nse pt validated:       '+str(gof.nse(np.array(df.pt_val),np.array(df.obs))))
file.write('nse pt calibrated:      '+str(gof.nse(np.array(df.pt_cal),np.array(df.obs))))
file.write('nse pt:                 '+str(gof.nse(np.array(df.pt),np.array(df.obs))))

file.write('\n#######################\n \n')

file.write('kge fao validated:     '+str(gof.kge(np.array(df.fao_cal),np.array(df.obs))))
file.write('kge fao calibrated:     '+str(gof.kge(np.array(df.fao_cal),np.array(df.obs))))
file.write('kge fao not calibrated: '+str(gof.kge(np.array(df.fao_not_cal),np.array(df.obs))))
file.write('kge ps calibrated:      '+str(gof.kge(np.array(df.ps_cal),np.array(df.obs))))
file.write('kge ps not calibrated:  '+str(gof.kge(np.array(df.ps_not_cal),np.array(df.obs))))
file.write('kge pt calibrated:      '+str(gof.kge(np.array(df.pt_cal),np.array(df.obs))))
file.write('kge pt not calibrated:  '+str(gof.kge(np.array(df.pt_not_cal),np.array(df.obs))))'''


file.write('#######################\n')
file.write('######   DAILY   ######\n')
file.write('#######################\n \n')

file.write('RMSE ps validated:          '+str(((dfDay.ps_val - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE ps calibrated:         '+str(((dfDay.ps_cal - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE ps:                    '+str(((dfDay.ps - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('\n')

file.write('RMSE fao validated:         '+str(((dfDay.fao_val - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE fao calibrated:        '+str(((dfDay.fao_cal - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE fao:                   '+str(((dfDay.fao - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('\n')

file.write('RMSE pt calibrated:         '+str(((dfDay.pt_val - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE pt calibrated:         '+str(((dfDay.pt_cal - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('RMSE pt:                    '+str(((dfDay.pt - dfDay.obs) ** 2).mean() ** .5)+'\n')
file.write('\n')

file.write('#######################\n \n')

file.write('MAE ps validated:           '+str(dfDay.eval('ps_val-obs').abs().mean())+'\n')
file.write('MAE ps calibrated:          '+str(dfDay.eval('ps_cal-obs').abs().mean())+'\n')
file.write('MAE ps:                     '+str(dfDay.eval('ps-obs').abs().mean())+'\n')
file.write('\n')

file.write('MAE fao validated:          '+str(dfDay.eval('fao_val-obs').abs().mean())+'\n')
file.write('MAE fao calibrated:         '+str(dfDay.eval('fao_cal-obs').abs().mean())+'\n')
file.write('MAE fao:                    '+str(dfDay.eval('fao-obs').abs().mean())+'\n')
file.write('\n')

file.write('MAE pt validated:           '+str(dfDay.eval('pt_val-obs').abs().mean())+'\n')
file.write('MAE pt calibrated:          '+str(dfDay.eval('pt_cal-obs').abs().mean())+'\n')
file.write('MAE pt:                     '+str(dfDay.eval('pt-obs').abs().mean())+'\n')
file.write('\n')

file.write('#######################\n \n')

file.write('R^{2} ps validated:         '+str(dfDay.ps_val.corr(dfDay.obs))+'\n')
file.write('R^{2} ps calibrated:        '+str(dfDay.ps_cal.corr(dfDay.obs))+'\n')
file.write('R^{2} ps:                   '+str(dfDay.ps.corr(dfDay.obs))+'\n')
file.write('\n')

file.write('R^{2} fao validated:        '+str(dfDay.fao_val.corr(dfDay.obs))+'\n')
file.write('R^{2} fao calibrated:       '+str(dfDay.fao_cal.corr(dfDay.obs))+'\n')
file.write('R^{2} fao:                  '+str(dfDay.fao.corr(dfDay.obs))+'\n')
file.write('\n')

file.write('R^{2} pt validated:         '+str(dfDay.pt_val.corr(dfDay.obs))+'\n')
file.write('R^{2} pt calibrated:        '+str(dfDay.pt_cal.corr(dfDay.obs))+'\n')
file.write('R^{2} pt:                   '+str(dfDay.pt.corr(dfDay.obs))+'\n')
file.write('\n')

file.write('#######################\n \n')

file.write('index d ps validated:       '+str(gof.d(np.array(dfDay.ps_val),np.array(dfDay.obs)))+'\n')
file.write('index d ps calibrated:      '+str(gof.d(np.array(dfDay.ps_cal),np.array(dfDay.obs)))+'\n')
file.write('index d ps:                 '+str(gof.d(np.array(dfDay.ps),np.array(dfDay.obs)))+'\n')
file.write('\n')

file.write('index d fao validated:      '+str(gof.d(np.array(dfDay.fao_val),np.array(dfDay.obs)))+'\n')
file.write('index d fao calibrated:     '+str(gof.d(np.array(dfDay.fao_cal),np.array(dfDay.obs)))+'\n')
file.write('index d fao:                '+str(gof.d(np.array(dfDay.fao),np.array(dfDay.obs)))+'\n')
file.write('\n')

file.write('index d pt validated:       '+str(gof.d(np.array(dfDay.pt_val),np.array(dfDay.obs)))+'\n')
file.write('index d pt calibrated:      '+str(gof.d(np.array(dfDay.pt_cal),np.array(dfDay.obs)))+'\n')
file.write('index d pt:                 '+str(gof.d(np.array(dfDay.pt),np.array(dfDay.obs)))+'\n')
file.write('\n')

file.write('#######################\n \n')

'''file.write('nse fao calibrated:     '+str(gof.nse(np.array(dfDay.fao_cal),np.array(dfDay.obs))))
file.write('nse fao calibrated:     '+str(gof.nse(np.array(dfDay.fao_cal),np.array(dfDay.obs))))
file.write('nse fao not calibrated: '+str(gof.nse(np.array(dfDay.fao_not_cal),np.array(dfDay.obs))))
file.write('nse ps calibrated:      '+str(gof.nse(np.array(dfDay.ps_cal),np.array(dfDay.obs))))
file.write('nse ps calibrated:      '+str(gof.nse(np.array(dfDay.ps_cal),np.array(dfDay.obs))))
file.write('nse ps not calibrated:  '+str(gof.nse(np.array(dfDay.ps_not_cal),np.array(dfDay.obs))))
file.write('nse pt calibrated:      '+str(gof.nse(np.array(dfDay.pt_cal),np.array(dfDay.obs))))
file.write('nse pt calibrated:      '+str(gof.nse(np.array(dfDay.pt_cal),np.array(dfDay.obs))))
file.write('nse pt not calibrated:  '+str(gof.nse(np.array(dfDay.pt_not_cal),np.array(dfDay.obs))))

file.write('\n#######################\n \n')

file.write('kge fao calibrated:     '+str(gof.kge(np.array(dfDay.fao_cal),np.array(dfDay.obs))))
file.write('kge fao calibrated:     '+str(gof.kge(np.array(dfDay.fao_cal),np.array(dfDay.obs))))
file.write('kge fao not calibrated: '+str(gof.kge(np.array(dfDay.fao_not_cal),np.array(dfDay.obs))))
file.write('kge ps calibrated:      '+str(gof.kge(np.array(dfDay.ps_cal),np.array(dfDay.obs))))
file.write('kge ps calibrated:      '+str(gof.kge(np.array(dfDay.ps_cal),np.array(dfDay.obs))))
file.write('kge ps not calibrated:  '+str(gof.kge(np.array(dfDay.ps_not_cal),np.array(dfDay.obs))))
file.write('kge pt calibrated:      '+str(gof.kge(np.array(dfDay.pt_cal),np.array(dfDay.obs))))
file.write('kge pt calibrated:      '+str(gof.kge(np.array(dfDay.pt_cal),np.array(dfDay.obs))))
file.write('kge pt not calibrated:  '+str(gof.kge(np.array(dfDay.pt_not_cal),np.array(dfDay.obs))))'''

file.write('\n#######################\n \n')
file.close()

dfMonth = dfDay[['obs','ps_val','ps_cal','ps','fao_val','fao_cal','fao','pt_val','pt_cal','pt']].copy()
dfYear = dfDay[['obs','ps_val','ps_cal','ps','fao_val','fao_cal','fao','pt_val','pt_cal','pt']].copy()

dfMonth = dfMonth.resample('1MS').agg('sum')
dfYear = dfYear.resample('1YS').agg('sum')

dfMonth['year'] = dfMonth.index.year
dfYear['year'] = dfYear.index.year
dfPrpY = pd.read_csv('/home/drugo/simProspero/data/'+def_site+'/'+def_site+'_PRP_YEAR.csv', index_col='TIMESTAMP')

#dfPrpY = dfPrpY.iloc[2:,:]
dfPrpY.columns = ['prp']
dfYear = pd.concat([dfYear,dfPrpY], axis=1)

'''
plt.style.use('default')
fig1 = plt.figure(figsize=(35,10))
ax1 = fig1.add_subplot(111)

dfMonth.plot(y=['obs',
                'ps_val','ps_cal','ps',
                'ps_val','fao_cal','fao',
                'ps_val','pt_cal','pt'],
             kind='line', ax = ax1, 
             colormap='Spectral',linewidth=5, rot = -0,stacked=False,
             label = ['Observed',
                      'PS validated','PS calibrated','PS uncalibrated',
                      'FAO validated','FAO calibrated','FAO uncalibrated',
                      'PT validated','PT calibrated','PT uncalibrated'])


ax1.set_ylabel('ET [mm]',fontsize=30,fontweight='bold')
ax1.set_xlabel('Year',fontsize=30,fontweight='bold')
ax1.legend(loc='upper center',fontsize=20, bbox_to_anchor=(1.075, 1),#ncol=11,
           handleheight=1, labelspacing=0.05,frameon=False)      

# axy.legend(loc='upper center',fontsize=30, bbox_to_anchor=(0.85, 1.15),ncol=2,handleheight=1, labelspacing=0.05)  
plt.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=30)
fig1.savefig('plots/plot_'+def_site+'.png')


plt.style.use('default')
fig2 = plt.figure(figsize=(35,10))
ax2 = fig2.add_subplot(111)
fig2.suptitle(def_site,fontsize=40)

dfYear.plot(y=['obs',
                'ps_val','ps_cal','ps',
                'ps_val','fao_cal','fao',
                'ps_val','pt_cal','pt',
                'prp'],
            x='year',kind='bar', ax = ax2, 
             colormap='Spectral',edgecolor=".3",linewidth=1.5, rot = -0,stacked=False,
             label = ['Observed',
                      'PS validated','PS calibrated','PS uncalibrated',
                      'FAO validated','FAO calibrated','FAO uncalibrated',
                      'PT validated','PT calibrated','PT uncalibrated',
                      'Precipitation'])


ax2.set_ylabel('ET [mm]',fontsize=30,fontweight='bold')
ax2.set_xlabel('Year',fontsize=30,fontweight='bold')
ax2.legend(loc='upper center',fontsize=20, bbox_to_anchor=(1.075, 1),#ncol=11,
           handleheight=1, labelspacing=0.05,frameon=False)   

# axy.legend(loc='upper center',fontsize=30, bbox_to_anchor=(0.85, 1.15),ncol=2,handleheight=1, labelspacing=0.05)  
plt.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=30)
fig2.savefig('plots/histogram_'+def_site+'.png')




fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharey=True)
fig.suptitle(def_site,fontsize=20)

sns.kdeplot(data=dfDay, ax=axes[0][0],x=dfDay.obs, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[0][1],x=dfDay.ps_val, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[0][2],x=dfDay.ps_cal, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[0][3],x=dfDay.ps, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[0][4],x=dfDay.fao_val, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[1][0],x=dfDay.fao_cal, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[1][1],x=dfDay.fao, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[1][2],x=dfDay.pt_val, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[1][3],x=dfDay.pt_cal, hue="SEASON")
sns.kdeplot(data=dfDay, ax=axes[1][4],x=dfDay.pt, hue="SEASON")
#axes[0].set_title('Observed')
axes[0][0].set_xlabel('Observed',fontsize=15)
axes[0][1].set_xlabel('PS validated',fontsize=15)
axes[0][2].set_xlabel('PS calibrated',fontsize=15)
axes[0][3].set_xlabel('PS uncalibrated',fontsize=15)
axes[0][4].set_xlabel('FAO validated',fontsize=15)
axes[1][0].set_xlabel('FAO calibrated',fontsize=15)
axes[1][1].set_xlabel('FAO uncalibrated',fontsize=15)
axes[1][2].set_xlabel('PT validated',fontsize=15)
axes[1][3].set_xlabel('PT calibrated',fontsize=15)
axes[1][4].set_xlabel('PT uncalibrated',fontsize=15)


for i in range(0,2):
    for j in range(0,5):
        #axes[i].legend().set_visible(False)
        #axes[i][j].set_ylim((0,1.2))
        #axes[i][j].set_xlim((-1,8))
        axes[i][j].set_ylabel('Density',fontsize=15)
        axes[i][j].tick_params(axis='both', which='major', labelsize=15)

fig.savefig('plots/distribution_'+def_site+'.png')'''

