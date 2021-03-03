#!/usr/bin/env python
# coding: utf-8

# # Data Analysis

import pandas as pd
import numpy as np
import os


os.chdir('FLX_AU-DaS_FLUXNET2015_FULLSET_2008-2014_2-4')
df = pd.read_csv("FLX_AU-DaS_FLUXNET2015_FULLSET_HH_2008-2014_2-4.csv", na_values = '-9999', parse_dates=[0,1], index_col='TIMESTAMP_START')

pd.set_option('display.max_columns', None)
site_name = 'AU_Das'
file=open(site_name+'_data_analysis.txt','a')

'''
dfnetta = df.LW_IN_F_MDS - df.LW_OUT + df.SW_IN_F_MDS - df.SW_OUT
dfnetta
df.NETRAD
dfdiff = dfnetta - df.NETRAD
dfdiff.plot()
dfnetta.plot(figsize = [10,5])
df.NETRAD.plot(figsize = [10,5])'''


# ### Precipitation
if 'P_F' in df:
    df_prp = df[['P_F','P_F_QC']]
    file.write('#############\n')
    file.write('Precipitation:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_prp.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_prp['P_F'].isna().sum())+'\n')

    file.write('Fraction of nan before cleaning'+str(df_prp['P_F'].isna().sum()*100/df.shape[0])+'\n')
    file.write('the max'+str(df_prp['P_F_QC'].max())+'\n')
    file.write('the min'+str(df_prp['P_F_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_prp['P_F_QC']==0))+'and %'+str(100*sum(df_prp['P_F_QC']==0)/df_prp.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_prp['P_F_QC']==1))+'and %'+str(100*sum(df_prp['P_F_QC']==1)/df_prp.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_prp['P_F_QC']==2))+'and %'+str(100*sum(df_prp['P_F_QC']==2)/df_prp.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_prp['P_F_QC']==3))+'and %'+str(100*sum(df_prp['P_F_QC']==3)/df_prp.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_prp.P_F_QC.max())+'\n')
    df_prp = df_prp.copy()
    df_prp[df_prp['P_F_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_prp.P_F_QC.max())+'\n')
    file.write('Fraction of nan after cleaning:'+str(df_prp['P_F_QC'].isna().sum()*100/df_prp.shape[0])+'\n')
    df_prp.to_csv(site_name+'_PRP.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np PRP value:')
    file.write('#############')

# ### Air Temperature
if 'TA_F_MDS' in df:
    df_ta = df[['TA_F_MDS','TA_F_MDS_QC']]
    file.write('#############\n')
    file.write('Air temperature:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_ta.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_ta['TA_F_MDS'].isna().sum())+'\n')
    file.write('Fraction of nan before cleaning'+str(df_ta['TA_F_MDS'].isna().sum()*100/df.shape[0])+'\n')
    file.write('the max'+str(df_ta['TA_F_MDS_QC'].max())+'\n')
    file.write('the min'+str(df_ta['TA_F_MDS_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_ta['TA_F_MDS_QC']==0))+'and %'+str(100*sum(df_ta['TA_F_MDS_QC']==0)/df_ta.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_ta['TA_F_MDS_QC']==1))+'and %'+str(100*sum(df_ta['TA_F_MDS_QC']==1)/df_ta.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_ta['TA_F_MDS_QC']==2))+'and %'+str(100*sum(df_ta['TA_F_MDS_QC']==2)/df_ta.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_ta['TA_F_MDS_QC']==3))+'and %'+str(100*sum(df_ta['TA_F_MDS_QC']==3)/df_ta.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_ta.TA_F_MDS_QC.max())+'\n')
    df_ta = df_ta.copy()
    df_ta[df_ta['TA_F_MDS_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_ta.TA_F_MDS_QC.max())+'\n')
    file.write('Fraction of nan after cleaning:'+str(df_ta['TA_F_MDS'].isna().sum()*100/df_ta.shape[0])+'\n')
    # df_ta.TA_F_MDS.plot()
    df_ta.to_csv(site_name+'_Temp.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np TA value:')
    file.write('#############')

# ### Net radiation
if 'NETRAD' in df:
    df_net = df[['NETRAD']]
    file.write('#############\n')
    file.write('Net Radiation:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_net.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_net['NETRAD'].isna().sum())+'\n')
    file.write('Fraction of nan before cleaning:'+str(df_net['NETRAD'].isna().sum()*100/df_net.shape[0])+'\n')
    # file.write('the max',df_net['SW_IN_F_MDS_QC'].max())
    # file.write('the min',df_net['SW_IN_F_MDS_QC'].min())
    # file.write('Total data QC equal to 0'+str(sum(df_net['SW_IN_F_MDS_QC']==0))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==0)/df_sw.shape[0])
    # file.write('Total data QC equal to 1'+str(sum(df_net['SW_IN_F_MDS_QC']==1))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==1)/df_sw.shape[0])
    # file.write('Total data QC equal to 2'+str(sum(df_sw['SW_IN_F_MDS_QC']==2))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==2)/df_sw.shape[0])
    # file.write('Total data QC equal to 3'+str(sum(df_sw['SW_IN_F_MDS_QC']==3))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==3)/df_sw.shape[0])
    # file.write('Maximum value before cleaning:',df_net.SW_IN_F_MDS_QC.max())
    # df_sw = df_sw.copy()
    # df_sw[df_sw['SW_IN_F_MDS_QC']>2]=np.nan
    # file.write('Maximum value before cleaning:',df_sw.SW_IN_F_MDS_QC.max())
    # file.write('Fraction of nan after cleaning:',df_sw['SW_IN_F_MDS'].isna().sum()*100/df_sw.shape[0])
    # df_sw.SW_IN_F_MDS.plot()
    df_net.to_csv(site_name+'_Net.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np NET value:')
    file.write('#############')


# ### Shortwave radiation In
if 'SW_IN_F_MDS' in df:
    df_sw = df[['SW_IN_F_MDS','SW_IN_F_MDS_QC']]
    file.write('#############\n')
    file.write('Shortwave in:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_sw.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_sw['SW_IN_F_MDS'].isna().sum())+'\n')
    file.write('Fraction of nan before cleaning:'+str(df_sw['SW_IN_F_MDS'].isna().sum()*100/df_sw.shape[0])+'\n')
    file.write('the max'+str(df_sw['SW_IN_F_MDS_QC'].max())+'\n')
    file.write('the min'+str(df_sw['SW_IN_F_MDS_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_sw['SW_IN_F_MDS_QC']==0))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==0)/df_sw.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_sw['SW_IN_F_MDS_QC']==1))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==1)/df_sw.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_sw['SW_IN_F_MDS_QC']==2))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==2)/df_sw.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_sw['SW_IN_F_MDS_QC']==3))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==3)/df_sw.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_sw.SW_IN_F_MDS_QC.max())+'\n')
    df_sw = df_sw.copy()
    df_sw[df_sw['SW_IN_F_MDS_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_sw.SW_IN_F_MDS_QC.max())+'\n')
    file.write('Fraction of nan after cleaning:'+str(df_sw['SW_IN_F_MDS'].isna().sum()*100/df_sw.shape[0])+'\n')
    # df_sw.SW_IN_F_MDS.plot()
    df_sw.to_csv(site_name+'_SwDirect.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np SW_IN value:')
    file.write('#############')


# ### Shortwave radiation Out
if 'SW_OUT' in df:
    df_sw_out = df[['SW_OUT']]
    file.write('#############\n')
    file.write('Shortwave out:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_sw_out.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_sw_out['SW_OUT'].isna().sum())+'\n')
    file.write('Fraction of nan before cleaning:'+str(df_sw_out['SW_OUT'].isna().sum()*100/df_sw_out.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_sw.SW_IN_F_MDS_QC.max())+'\n')
    df_sw = df_sw.copy()
    df_sw[df_sw['SW_IN_F_MDS_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_sw.SW_IN_F_MDS_QC.max())+'\n')
    file.write('Fraction of nan after cleaning:'+str(df_sw['SW_IN_F_MDS'].isna().sum()*100/df_sw.shape[0])+'\n')
    df_sw.to_csv(site_name+'_Swout.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np SW_OUT value:')
    file.write('#############')



'''
# ### Shortwave radiation In
if 'TA_F_MDS' in df:
df_sw = df[['SW_IN_F_MDS','SW_IN_F_MDS_QC']]
file.write('Data:',df_sw.shape)
file.write('Total nan before cleaning:',df_sw['SW_IN_F_MDS'].isna().sum())
file.write('Fraction of nan before cleaning:',df_sw['SW_IN_F_MDS'].isna().sum()*100/df_sw.shape[0])
file.write('the max',df_sw['SW_IN_F_MDS_QC'].max())
file.write('the min',df_sw['SW_IN_F_MDS_QC'].min())

file.write('Total data QC equal to 0'+str(sum(df_sw['SW_IN_F_MDS_QC']==0))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==0)/df_sw.shape[0])
file.write('Total data QC equal to 1'+str(sum(df_sw['SW_IN_F_MDS_QC']==1))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==1)/df_sw.shape[0])
file.write('Total data QC equal to 2'+str(sum(df_sw['SW_IN_F_MDS_QC']==2))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==2)/df_sw.shape[0])
file.write('Total data QC equal to 3'+str(sum(df_sw['SW_IN_F_MDS_QC']==3))+'and %'+str(100*sum(df_sw['SW_IN_F_MDS_QC']==3)/df_sw.shape[0])

file.write('Maximum value before cleaning:',df_sw.SW_IN_F_MDS_QC.max())
df_sw = df_sw.copy()
df_sw[df_sw['SW_IN_F_MDS_QC']>2]=np.nan
file.write('Maximum value before cleaning:',df_sw.SW_IN_F_MDS_QC.max())
file.write('Fraction of nan after cleaning:',df_sw['SW_IN_F_MDS'].isna().sum()*100/df_sw.shape[0])
# df_sw.SW_IN_F_MDS.plot()
df_sw.to_csv('swRadUSVar.csv',na_rep=-9999)'''


# ### Longwave radiation
if 'LW_IN_F_MDS' in df:
    df_lw = df[['LW_IN_F_MDS', 'LW_IN_F_MDS_QC']]
    file.write('#############\n')
    file.write('Longwave:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_lw.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_lw['LW_IN_F_MDS'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_lw['LW_IN_F_MDS'].isna().sum()*100/df_lw.shape[0])+'\n')
    file.write('the max'+str(df_lw['LW_IN_F_MDS_QC'].max())+'\n')
    file.write('the min'+str(df_lw['LW_IN_F_MDS_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_lw['LW_IN_F_MDS_QC']==0))+'and %'+str(100*sum(df_lw['LW_IN_F_MDS_QC']==0)/df_lw.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_lw['LW_IN_F_MDS_QC']==1))+'and %'+str(100*sum(df_lw['LW_IN_F_MDS_QC']==1)/df_lw.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_lw['LW_IN_F_MDS_QC']==2))+'and %'+str(100*sum(df_lw['LW_IN_F_MDS_QC']==2)/df_lw.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_lw['LW_IN_F_MDS_QC']==3))+'and %'+str(100*sum(df_lw['LW_IN_F_MDS_QC']==3)/df_lw.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_lw.LW_IN_F_MDS_QC.max())+'\n')
    df_lw = df_lw.copy()
    df_lw[df_lw['LW_IN_F_MDS_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_lw.LW_IN_F_MDS_QC.max())+'\n')
    # df_lw.LW_IN_F_MDS.plot()
    df_lw.to_csv(site_name+'_LwRad.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np LW_IN_F_MDS value:')
    file.write('#############')

# ### Atmospheric pressure
if 'PA_F' in df:
    df_pa = df[['PA_F', 'PA_F_QC']]
    file.write('#############\n')
    file.write('Pressure:\n')
    file.write('#############\n')
    file.write('Data:'+str(df_pa.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_pa['PA_F'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_pa['PA_F'].isna().sum()*100/df_pa.shape[0])+'\n')
    file.write('the max'+str(df_pa['PA_F_QC'].max())+'\n')
    file.write('the min'+str(df_pa['PA_F_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_pa['PA_F_QC']==0))+'and %'+str(100*sum(df_pa['PA_F_QC']==0)/df_pa.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_pa['PA_F_QC']==1))+'and %'+str(100*sum(df_pa['PA_F_QC']==1)/df_pa.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_pa['PA_F_QC']==2))+'and %'+str(100*sum(df_pa['PA_F_QC']==2)/df_pa.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_pa['PA_F_QC']==3))+'and %'+str(100*sum(df_pa['PA_F_QC']==3)/df_pa.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_pa.PA_F_QC.max())+'\n')
    df_pa = df_pa.copy()
    df_pa[df_pa['PA_F_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_pa.PA_F_QC.max())+'\n')
    # df_pa.PA_F.plot()
    df_pa.to_csv(site_name+'_Pres.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np PA_F value:')
    file.write('#############')


# ### Wind speed
if 'WS_F' in df:
    df_ws = df[['WS_F', 'WS_F_QC']]
    file.write('#############\n')
    file.write('Wind speed:\n')
    file.write('#############\n')
    # df_ws.WS_F.plot()
    file.write('Data:'+str(df_ws.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_ws['WS_F'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_ws['WS_F'].isna().sum()*100/df_ws.shape[0])+'\n')
    file.write('the max'+str(df_ws['WS_F_QC'].max())+'\n')
    file.write('the min'+str(df_ws['WS_F_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_ws['WS_F_QC']==0))+'and %'+str(100*sum(df_ws['WS_F_QC']==0)/df_ws.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_ws['WS_F_QC']==1))+'and %'+str(100*sum(df_ws['WS_F_QC']==1)/df_ws.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_ws['WS_F_QC']==2))+'and %'+str(100*sum(df_ws['WS_F_QC']==2)/df_ws.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_ws['WS_F_QC']==3))+'and %'+str(100*sum(df_ws['WS_F_QC']==3)/df_ws.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_ws.WS_F_QC.max())+'\n')
    df_ws = df_ws.copy()
    df_ws[df_ws['WS_F_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_ws.WS_F_QC.max())+'\n')
    # df_ws.WS_F.plot()
    df_ws.to_csv(site_name+'_Wind.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np WS_F value:')
    file.write('#############')


# ### Vapor pressure deficit
if 'VPD_F' in df:
    df_vpd = df[['VPD_F' ,'VPD_F_QC']]
    file.write('#############\n')
    file.write('Vapour pressure deficit:\n')
    file.write('#############\n')
    # df_vpd.plot()
    file.write('Data:'+str(df_vpd.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_vpd['VPD_F'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_vpd['VPD_F'].isna().sum()*100/df_vpd.shape[0])+'\n')
    file.write('the max'+str(df_vpd['VPD_F_QC'].max())+'\n')
    file.write('the min'+str(df_vpd['VPD_F_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_vpd['VPD_F_QC']==0))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==0)/df_vpd.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_vpd['VPD_F_QC']==1))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==1)/df_vpd.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_vpd['VPD_F_QC']==2))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==2)/df_vpd.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_vpd['VPD_F_QC']==3))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==3)/df_vpd.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_vpd.VPD_F_QC.max())+'\n')
    df_vpd = df_vpd.copy()
    df_vpd[df_vpd['VPD_F_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_vpd.VPD_F_QC.max())+'\n')
    # df_vpd.VPD_F.plot()
    df_vpd.to_csv(site_name+'_VDP.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np VPD_F value:')
    file.write('#############')


# ### Soil heat flux
if 'G_F_MDS' in df:
    file.write('#############\n')
    file.write('Ground heat flux:\n')
    file.write('#############\n')
    df_gh = df[['G_F_MDS' ,'G_F_MDS_QC']]
    # df_gh.plot()
    file.write('Data:'+str(df_gh.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_gh['G_F_MDS'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_gh['G_F_MDS'].isna().sum()*100/df_gh.shape[0])+'\n')
    file.write('the max'+str(df_gh['G_F_MDS_QC'].max())+'\n')
    file.write('the min'+str(df_gh['G_F_MDS_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_gh['G_F_MDS_QC']==0))+'and %'+str(100*sum(df_gh['G_F_MDS_QC']==0)/df_gh.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_gh['G_F_MDS_QC']==1))+'and %'+str(100*sum(df_gh['G_F_MDS_QC']==1)/df_gh.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_gh['G_F_MDS_QC']==2))+'and %'+str(100*sum(df_gh['G_F_MDS_QC']==2)/df_gh.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_gh['G_F_MDS_QC']==3))+'and %'+str(100*sum(df_gh['G_F_MDS_QC']==3)/df_gh.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_gh.G_F_MDS_QC.max())+'\n')
    df_gh = df_gh.copy()
    df_gh[df_gh['G_F_MDS_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_gh.G_F_MDS_QC.max())+'\n')
    # df_gh.G_F_MDS.plot()
    df_gh.to_csv(site_name+'_GHF.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np G_F_MDS value:')
    file.write('#############')
    
# ### Latent heat
if 'LE_CORR' in df:
    file.write('#############\n')
    file.write('Latent heat:\n')
    file.write('#############\n')
    df_le = df[['LE_CORR','LE_CORR_25','LE_CORR_75','LE_F_MDS','LE_RANDUNC','LE_F_MDS_QC']]
    # df_le.plot()
    file.write('Data:'+str(df_le.shape)+'\n')
    file.write('Total LE_F_MDS nan before cleaning:'+str(df_le['LE_F_MDS'].isna().sum())+'\n')
    file.write('Total LE_CORR nan before cleaning:'+str(df_le['LE_CORR'].isna().sum())+'\n')
    file.write('Total LE_CORR_25 nan before cleaning:'+str(df_le['LE_CORR_25'].isna().sum())+'\n')
    file.write('Total LE_CORR_75 nan before cleaning:'+str(df_le['LE_CORR_75'].isna().sum())+'\n')
    file.write('Total LE_RANDUNC nan before cleaning:'+str(df_le['LE_RANDUNC'].isna().sum())+'\n')
    file.write('Fraction of nan LE_F_MDS'+str(df_le['LE_F_MDS'].isna().sum()*100/df_le.shape[0])+'\n')
    file.write('Fraction of nan LE_CORR'+str(df_le['LE_CORR'].isna().sum()*100/df_le.shape[0])+'\n')
    file.write('Fraction of nan LE_CORR_25'+str(df_le['LE_CORR_25'].isna().sum()*100/df_le.shape[0])+'\n')
    file.write('Fraction of nan LE_CORR_75'+str(df_le['LE_CORR_75'].isna().sum()*100/df_le.shape[0])+'\n')
    file.write('the max'+str(df_le['LE_F_MDS_QC'].max())+'\n')
    file.write('the min'+str(df_le['LE_F_MDS_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_le['LE_F_MDS_QC']==0))+'and %'+str(100*sum(df_le['LE_F_MDS_QC']==0)/df_le.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_le['LE_F_MDS_QC']==1))+'and %'+str(100*sum(df_le['LE_F_MDS_QC']==1)/df_le.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_le['LE_F_MDS_QC']==2))+'and %'+str(100*sum(df_le['LE_F_MDS_QC']==2)/df_le.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_le['LE_F_MDS_QC']==3))+'and %'+str(100*sum(df_le['LE_F_MDS_QC']==3)/df_le.shape[0])+'\n')
    file.write(str(df_le['LE_F_MDS'].max())+'\n')
    file.write(str(df_le['LE_CORR'].max())+'\n')
    file.write(str(df_le['LE_CORR_25'].max())+'\n')
    file.write(str(df_le['LE_CORR_75'].max())+'\n')
    file.write(str(df_le['LE_F_MDS'].mean()))
    file.write(str(df_le['LE_CORR'].mean()))
    file.write(str(df_le['LE_CORR_25'].mean()))
    file.write(str(df_le['LE_CORR_75'].mean()))
    file.write('Maximum value before cleaning:'+str(df_le.LE_F_MDS_QC.max())+'\n')
    df_le = df_le.copy()
    df_le[df_le['LE_F_MDS_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_le.LE_F_MDS_QC.max())+'\n')
    # df_le.LE_CORR.plot()
    # df_le.LE_F_MDS.plot()
    df_le[['LE_CORR']].to_csv(site_name+'_ElCorr.csv',na_rep=-9999)
    df_le[['LE_F_MDS']].to_csv(site_name+'_El.csv',na_rep=-9999)
    df_le.head(48)
else:
    file.write('#############')
    file.write('Np LE_CORR value:')
    file.write('#############')

# ### Sensible heat
if 'H_CORR' in df:
    file.write('#############\n')
    file.write('Sensible heat:\n')
    file.write('#############\n')
    df_hl = df[['H_CORR','H_CORR_25','H_CORR_75','H_F_MDS','H_RANDUNC','H_F_MDS_QC']]
    # df_hl.plot()
    file.write('Data:'+str(df_hl.shape)+'\n')
    file.write('Total H_F_MDS nan before cleaning:'+str(df_hl['H_F_MDS'].isna().sum())+'\n')
    file.write('Total H_CORR nan before cleaning:'+str(df_hl['H_CORR'].isna().sum())+'\n')
    file.write('Total H_CORR_25 nan before cleaning:'+str(df_hl['H_CORR_25'].isna().sum())+'\n')
    file.write('Total H_CORR_75 nan before cleaning:'+str(df_hl['H_CORR_75'].isna().sum())+'\n')
    file.write('Total H_RANDUNC nan before cleaning:'+str(df_hl['H_RANDUNC'].isna().sum())+'\n')
    file.write('Fraction of nan H_F_MDS'+str(df_hl['H_F_MDS'].isna().sum()*100/df_hl.shape[0])+'\n')
    file.write('Fraction of nan H_CORR'+str(df_hl['H_CORR'].isna().sum()*100/df_hl.shape[0])+'\n')
    file.write('Fraction of nan H_CORR_25'+str(df_hl['H_CORR_25'].isna().sum()*100/df_hl.shape[0])+'\n')
    file.write('Fraction of nan H_CORR_75'+str(df_hl['H_CORR_75'].isna().sum()*100/df_hl.shape[0])+'\n')
    file.write('the max'+str(df_hl['H_F_MDS_QC'].max())+'\n')
    file.write('the min'+str(df_hl['H_F_MDS_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_hl['H_F_MDS_QC']==0))+'and %'+str(100*sum(df_hl['H_F_MDS_QC']==0)/df_hl.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_hl['H_F_MDS_QC']==1))+'and %'+str(100*sum(df_hl['H_F_MDS_QC']==1)/df_hl.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_hl['H_F_MDS_QC']==2))+'and %'+str(100*sum(df_hl['H_F_MDS_QC']==2)/df_hl.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_hl['H_F_MDS_QC']==3))+'and %'+str(100*sum(df_hl['H_F_MDS_QC']==3)/df_hl.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_hl.H_F_MDS_QC.max())+'\n')
    df_hl = df_hl.copy()
    df_hl[df_hl['H_F_MDS_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_hl.H_F_MDS_QC.max())+'\n')
    df_hl[['H_CORR']].to_csv(site_name+'_HlCorr.csv',na_rep=-9999)
    df_hl[['H_F_MDS']].to_csv(site_name+'_Hl.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np H_CORR value:')
    file.write('#############')
df.RH


# ### Relative Humidity
if 'RH' in df:
    file.write('#############\n')
    file.write('Relative humidity:\n')
    file.write('#############\n')    
    df_rh = df[['RH']]
    file.write('Data:'+str(df_rh.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_rh['RH'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_rh['RH'].isna().sum()*100/df_rh.shape[0])+'\n')
    # file.write('the max',df_rh['VPD_F_QC'].max())
    # file.write('the min',df_rh['VPD_F_QC'].min())
    # file.write('Total data QC equal to 0'+str(sum(df_vpd['VPD_F_QC']==0))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==0)/df_vpd.shape[0])
    # file.write('Total data QC equal to 1'+str(sum(df_vpd['VPD_F_QC']==1))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==1)/df_vpd.shape[0])
    # file.write('Total data QC equal to 2'+str(sum(df_vpd['VPD_F_QC']==2))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==2)/df_vpd.shape[0])
    # file.write('Total data QC equal to 3'+str(sum(df_vpd['VPD_F_QC']==3))+'and %'+str(100*sum(df_vpd['VPD_F_QC']==3)/df_vpd.shape[0])
    # file.write('Maximum value before cleaning:',df_vpd.VPD_F_QC.max())
    # df_vpd = df_vpd.copy()
    # df_vpd[df_vpd['VPD_F_QC']>2]=np.nan
    # file.write('Maximum value before cleaning:',df_vpd.VPD_F_QC.max())
    # df_ta.TA_F_MDS.plot()
    df_rh.to_csv(site_name+'_RH.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np RH value:')
    file.write('#############')

# ### Soil water content
if 'SWC_F_MDS_1' in df:
    file.write('#############\n')
    file.write('Soil water content 1:\n')
    file.write('#############\n')
    df_swc = df[['SWC_F_MDS_1', 'SWC_F_MDS_1_QC']]
    df_swc.head(10)
    file.write('Data:'+str(df_swc.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_swc['SWC_F_MDS_1'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_swc['SWC_F_MDS_1'].isna().sum()*100/df_swc.shape[0])+'\n')
    file.write('the max'+str(df_swc['SWC_F_MDS_1_QC'].max())+'\n')
    file.write('the min'+str(df_swc['SWC_F_MDS_1_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_swc['SWC_F_MDS_1_QC']==0))+'and %'+str(100*sum(df_swc['SWC_F_MDS_1_QC']==0)/df_swc.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_swc['SWC_F_MDS_1_QC']==1))+'and %'+str(100*sum(df_swc['SWC_F_MDS_1_QC']==1)/df_swc.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_swc['SWC_F_MDS_1_QC']==2))+'and %'+str(100*sum(df_swc['SWC_F_MDS_1_QC']==2)/df_swc.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_swc['SWC_F_MDS_1_QC']==3))+'and %'+str(100*sum(df_swc['SWC_F_MDS_1_QC']==3)/df_swc.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_swc.SWC_F_MDS_1_QC.max())+'\n')
    df_swc = df_swc.copy()
    df_swc[df_swc['SWC_F_MDS_1_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_swc.SWC_F_MDS_1_QC.max())+'\n')
    file.write('Fraction of nan after cleaning:'+str(df_swc['SWC_F_MDS_1_QC'].isna().sum()*100/df_swc.shape[0])+'\n')
    df_swc.SWC_F_MDS_1.plot()
    df_swc.to_csv(site_name+'_SWC.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np SWC_F_MDS_1 value:')
    file.write('#############')

# ### Soil water content
if 'SWC_F_MDS_2' in df:
    file.write('#############\n')
    file.write('Soil water content 2:\n')
    file.write('#############\n')    
    df_swc = df[['SWC_F_MDS_2', 'SWC_F_MDS_2_QC']]
    df_swc.head(10)
    file.write('Data:'+str(df_swc.shape)+'\n')
    file.write('Total nan before cleaning:'+str(df_swc['SWC_F_MDS_2'].isna().sum())+'\n')
    file.write('Fraction of nan'+str(df_swc['SWC_F_MDS_2'].isna().sum()*100/df_swc.shape[0])+'\n')
    file.write('the max'+str(df_swc['SWC_F_MDS_2_QC'].max())+'\n')
    file.write('the min'+str(df_swc['SWC_F_MDS_2_QC'].min())+'\n')
    file.write('Total data QC equal to 0'+str(sum(df_swc['SWC_F_MDS_2_QC']==0))+'and %'+str(100*sum(df_swc['SWC_F_MDS_2_QC']==0)/df_swc.shape[0])+'\n')
    file.write('Total data QC equal to 1'+str(sum(df_swc['SWC_F_MDS_2_QC']==1))+'and %'+str(100*sum(df_swc['SWC_F_MDS_2_QC']==1)/df_swc.shape[0])+'\n')
    file.write('Total data QC equal to 2'+str(sum(df_swc['SWC_F_MDS_2_QC']==2))+'and %'+str(100*sum(df_swc['SWC_F_MDS_2_QC']==2)/df_swc.shape[0])+'\n')
    file.write('Total data QC equal to 3'+str(sum(df_swc['SWC_F_MDS_2_QC']==3))+'and %'+str(100*sum(df_swc['SWC_F_MDS_2_QC']==3)/df_swc.shape[0])+'\n')
    file.write('Maximum value before cleaning:'+str(df_swc.SWC_F_MDS_2_QC.max())+'\n')
    df_swc = df_swc.copy()
    df_swc[df_swc['SWC_F_MDS_2_QC']>2]=np.nan
    file.write('Maximum value before cleaning:'+str(df_swc.SWC_F_MDS_2_QC.max())+'\n')
    file.write('Fraction of nan after cleaning:'+str(df_swc['SWC_F_MDS_2_QC'].isna().sum()*100/df_swc.shape[0])+'\n')
    df_swc.SWC_F_MDS_2.plot()
    df_swc.to_csv(site_name+'_SWC_2.csv',na_rep=-9999)
else:
    file.write('#############')
    file.write('Np SWC_F_MDS_2 value:')
    file.write('#############')


file.close()