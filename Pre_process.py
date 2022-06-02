import os
# print(os.getcwd())
import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime

AMOC_model = scipy.io.loadmat('AMOC_model.mat')

##### Obs data: get monthly data, detrend monthly mean ##################################
Obs = AMOC_model['MOC_mar_hc10']
Obs[Obs == -99999] = np.nan

data_Obs = {'Time_Obs': AMOC_model['Time_Obs'], 'Obs': Obs.reshape((10535,))}
data_Obs = pd.DataFrame(data_Obs)
data_Obs['Time_Obs'] = pd.to_datetime(data_Obs['Time_Obs'])
data_Obs = data_Obs.set_index(['Time_Obs'], drop=False)

year = data_Obs['Time_Obs'].dt.year
month = data_Obs['Time_Obs'].dt.month
index = []
for i in range(10535):
    index.append(str(year[i]) + '-' + str(month[i]))
norep_index = list(set(index))

month_mean = []
for i in range(2004, 2019):
    for j in range(1, 13):
        key = str(i) + '-' + str(j)
        if key in norep_index:
            month_mean.append([key, data_Obs.loc[key, 'Obs'].mean()])

data_Obs_monthly = pd.DataFrame(month_mean, columns=['Time_Obs', 'Obs'])

pre_data = []
for i in range(1957, 2019):
    for j in range(1, 13):
        key = str(i) + '-' + str(j)
        pre_data.append([key, np.nan])
pre_data = pd.DataFrame(pre_data, columns=['Time_Obs', 'Obs'])

pre_data.iloc[9, 1] = 20.1
pre_data.iloc[295, 1] = 17.3
pre_data.iloc[296, 1] = 17.3
pre_data.iloc[426, 1] = 18.5
pre_data.iloc[427, 1] = 18.5
pre_data.iloc[493, 1] = 18.1

pre_data.iloc[567:741, 1] = data_Obs_monthly['Obs']

pre_data['Time_Obs'] = pd.to_datetime(pre_data['Time_Obs'])

for i in range(1, 13):
    pre_data.loc[pre_data['Time_Obs'].dt.month == i, 'Obs'] = pre_data.loc[pre_data['Time_Obs'].dt.month == i, 'Obs'] - pre_data.loc[pre_data['Time_Obs'].dt.month == i, 'Obs'].mean()

dtdata_Obs = pre_data.iloc[9:740].reset_index(drop=True)

##########Ensenble Data, detrend monthly mean (month mean get by PD)###################################################################################
PD = AMOC_model['PD']
data = {'Time_PD': AMOC_model['Time_PD'], 'PD': PD.reshape((10788,))}
data_PD = pd.DataFrame(data)

month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

PD_mean = {}
for i in month:
    PD_mean[i] = data_PD.loc[data_PD['Time_PD'].str.contains(i), 'PD'].mean()

year = list(range(2001, 2500))
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Time_Ensemble = []
for i in year:
    for j in month:
        Time_Ensemble.append(str(i) + '-' + str(j))

Ensemble = {'Time_Ensemble': Time_Ensemble}
for i in range(1, 29):
    Ensemble[i] = AMOC_model['Ensemble'][:, i-1]

data_Ensemble = pd.DataFrame(Ensemble)

for i in month:
    data_Ensemble.loc[data_Ensemble['Time_Ensemble'].str.contains(i), range(1, 29)] = data_Ensemble.loc[data_Ensemble['Time_Ensemble'].str.contains(i), range(1, 29)] - PD_mean[i]

dtdata_Ensemble = data_Ensemble

os.chdir('C:\\Users\\yizic\\Documents\\3_LSTM\\VAE\\add_AMOCindex')
dtdata_Obs.to_csv(r'dtdata_Obs.csv')
dtdata_Ensemble.to_csv(r'dtdata_Ensemble.csv')

### check date_time label from .nc file #########################
from scipy.io import netcdf

AMOC = netcdf.NetCDFFile('AMOC_ENSEMBLE_monthly.nc','r')
time = AMOC.variables['time']
data = time[:]*1

VAR_265N = AMOC.variables['VAR_265N']
VAR_265N = VAR_265N[:]*1

VAR_28N = AMOC.variables['VAR_28N']
VAR_28N = VAR_28N[:]*1

VAR_48N = AMOC.variables['VAR_48N']
VAR_48N = VAR_48N[:]*1

VAR_MAX = AMOC.variables['VAR_MAX']
VAR_MAX = VAR_MAX[:]*1