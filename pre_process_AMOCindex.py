import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

os.chdir('C:\\Users\\yizic\\Documents\\3_LSTM\\VAE\\add_AMOCindex')

####Ensemble AMOC_index Data##############################################################################
year = list(range(2001, 2500))
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Time_Ensemble = []
for i in year:
    for j in month:
        Time_Ensemble.append(str(i) + '-' + str(j))

Ensemble = {'Time_Ensemble': Time_Ensemble}

for i in range(1, 29):
    words = ['input\\ENSEMBLE_r', str(i), '.txt']
    file = ''.join(words)
    a = pd.read_csv(file, sep=" ", header=None)
    Ensemble[i] = np.array(a).reshape((5988,))

data_Ensemble = pd.DataFrame(Ensemble)

######demean Ensemble with PD monthly mean###########################################################
year = list(range(501, 1400))
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Time_PD = []
for i in year:
    for j in month:
        Time_PD.append(str(i) + '-' + str(j))

PD = np.array(pd.read_csv('input\\PD_r1.txt', sep=" ", header=None))
data = {'Time_PD': Time_PD, 'PD': PD.reshape((10788,))}
data_PD = pd.DataFrame(data)

PD_mean = {}
for i in month:
    PD_mean[i] = data_PD.loc[data_PD['Time_PD'].str.contains(i), 'PD'].mean()

for i in month:
    data_Ensemble.loc[data_Ensemble['Time_Ensemble'].str.contains(i), range(1, 29)] = data_Ensemble.loc[data_Ensemble['Time_Ensemble'].str.contains(i), range(1, 29)] - PD_mean[i]

####Obs AMOC_index Data##############################################################################
year = list(range(1880, 2020))
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Time_Obs = []
for i in year:
    for j in month:
        Time_Obs.append(str(i) + '-' + str(j))

Obs = np.array(pd.read_csv('input\\obs_r1.txt', sep=" ", header=None)).reshape((1680,))
data = {'Time_Obs': Time_Obs, 'Obs': Obs}
data_Obs = pd.DataFrame(data)

Obs_mean = {}
for i in month:
    Obs_mean[i] = data_Obs.loc[data_Obs['Time_Obs'].str.contains(i), 'Obs'].mean()

for i in month:
    data_Obs.loc[data_Obs['Time_Obs'].str.contains(i), 'Obs'] = data_Obs.loc[data_Obs['Time_Obs'].str.contains(i), 'Obs'] - Obs_mean[i]

#########input data modification########################################################################################################
# data_Ensemble = data_Ensemble.iloc[0:5964, :]
#
dtdata_Obs = pd.read_csv("dtdata_Obs.csv", usecols=[1, 2])
dtdata_Ensemble = pd.read_csv("dtdata_Ensemble.csv", usecols=list(range(1, 30)))
#
# dtdata_Ensemble = dtdata_Ensemble.iloc[16:5983, :].reset_index(drop=True)
# Time_Ensemble = dtdata_Ensemble.loc[:, 'Time_Ensemble']
# year = list(range(2001, 2498))
# year = [str(i) for i in year]
#
# for i in year:
#     a = Time_Ensemble[Time_Ensemble.str.contains(i)]
#     if len(a) != 12:
#         print(i)
# # 2124, 2248, 2376 have 13 obs
#
# dtdata_Ensemble = dtdata_Ensemble.drop([1485, 2977, 4509]).reset_index(drop=True)

data = data_Obs.copy()
data.loc[:, 'Obs'] = np.nan
data.loc[933:1663, 'Obs'] = np.array(dtdata_Obs.loc[:, 'Obs'])
dtdata_Obs = data

data_Obs.to_csv(r'dtdata_AMOCindex_Obs.csv')
data_Ensemble.to_csv(r'dtdata_AMOCindex_Ensemble.csv')

dtdata_Obs.to_csv(r'dtdata_AMOC_Obs.csv')
dtdata_Ensemble.to_csv(r'dtdata_AMOC_Ensemble.csv')

# plt.figure(figsize=(9, 5))
# # plt.plot(data_Obs.iloc[:, 1], color='red', label='Obs')
# plt.plot(data_Ensemble.iloc[0:4000, 1], color='blue', label='Ensemble')
# plt.legend()

plt.figure(figsize=(9, 5))
# plt.plot(data_Obs.iloc[:, 1], color='red', label='Obs')
for i in range(28):
    plt.plot(sdata_AMOC[:, i], color='grey')
plt.show()