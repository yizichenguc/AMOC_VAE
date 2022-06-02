import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neighbors import KernelDensity


os.chdir('/Users/yizicheng/Documents/3_LSTM/pred')

##Data preprocessing############################################################

#X
X = {'Time': list(range(1850, 2015))}

for i in range(1, 13):
    words = ['Yizi/M', str(i), '_1850_2014_pp_Annual_AMOChistoricalmax265_clim.txt']
    file = ''.join(words)
    a = np.loadtxt(file)
    X[i] = a

X = pd.DataFrame(X)

#Y
Y = {'Time': list(range(2015, 2101))}

for i in range(1, 13):
    words = ['Yizi/M', str(i), '_2015_2100_pp_Annual_AMOCssp585max265_clim.txt']
    file = ''.join(words)
    a = np.loadtxt(file)
    Y[i] = a

Y = pd.DataFrame(Y)

# #plot
# plt.figure(figsize=(9, 5))
# for i in range(1, 13):
#     plt.plot(X.iloc[:, i], color='red')
# plt.show()

#Observation
Obs_pred = np.load('Obs_pred.npy')

Obs_pred_yearly = np.zeros(shape=(1000, 140))

for i in range(140):
    a = i*12
    Obs_pred_yearly[:, i] = np.mean(Obs_pred[:, a:(a+12), 0], axis=1)

Obs_pred_yearly_1 = np.reshape(np.ravel(Obs_pred_yearly, order='C'), (140, 1000), order='F')

Obs = pd.DataFrame(Obs_pred_yearly_1)
Obs.insert(0, 'Time', list(range(1880, 2020)))

# #fit linear regression for 12 model AMOC difference
# mean_1 = X[(X['Time'] >= 1880) & (X['Time'] <= 1919)].mean(axis=0)[1:13]
# mean_2 = X[(X['Time'] >= 1995) & (X['Time'] <= 2014)].mean(axis=0)[1:13]
# mean_3 = Y[(Y['Time'] >= 2080) & (Y['Time'] <= 2099)].mean(axis=0)[1:13]

#fit linear regression for 12 model AMOC difference: 40 years average
mean_1 = X[(X['Time'] >= 1880) & (X['Time'] <= 1919)].mean(axis=0)[1:13]
mean_2 = X[(X['Time'] >= 1975) & (X['Time'] <= 2014)].mean(axis=0)[1:13]
mean_3 = Y[(Y['Time'] >= 2060) & (Y['Time'] <= 2099)].mean(axis=0)[1:13]

delta_X = np.array(mean_2 - mean_1).reshape((-1, 1))
delta_Y = np.array(mean_3 - mean_2).reshape((-1, 1))

lr = linear_model.LinearRegression()
lr.fit(delta_X, delta_Y)

x = np.linspace(-2, 1.5, 500).reshape((-1,1))
y_pred = lr.predict(x)

# #get prediction for 1000 trails Observation data
# mean_1_Obs = Obs[(Obs['Time'] >= 1880) & (Obs['Time'] <= 1919)].mean(axis=0)[1:1001]
# mean_2_Obs = Obs[(Obs['Time'] >= 1995) & (Obs['Time'] <= 2014)].mean(axis=0)[1:1001]
#get prediction for 1000 trails Observation data, 40 years average
mean_1_Obs = Obs[(Obs['Time'] >= 1880) & (Obs['Time'] <= 1919)].mean(axis=0)[1:1001]
mean_2_Obs = Obs[(Obs['Time'] >= 1975) & (Obs['Time'] <= 2014)].mean(axis=0)[1:1001]
Obs_inp = np.array(mean_2_Obs - mean_1_Obs).reshape((-1, 1))
Obs_out = lr.predict(Obs_inp)

#input gaussian density estimation
kde_inp = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Obs_inp)
x_Obs_inp = np.linspace(np.min(Obs_inp), np.max(Obs_inp), 5000)[:, np.newaxis]
log_dens_x_inp = kde_inp.score_samples(x_Obs_inp)[:, np.newaxis]

#output gaussian density estimation
kde_out = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(Obs_out)
x_Obs_out = np.linspace(np.min(Obs_out), np.max(Obs_out), 5000)[:, np.newaxis]
log_dens_x_out = kde_out.score_samples(x_Obs_out)[:, np.newaxis]

a = np.where(log_dens_x_inp == np.max(log_dens_x_inp))
b = np.where(log_dens_x_out == np.max(log_dens_x_out))
#856
#4179

x_min = min(np.min(delta_X), np.min(Obs_inp))
x_max = max(np.max(delta_X), np.max(Obs_inp))
y_min = min(np.min(delta_Y), np.min(Obs_out))
y_max = max(np.max(delta_Y), np.max(Obs_out))

#plot
fig, axes = plt.subplots(2, 2, figsize=(9, 9))

axes[-1, -1].axis('off')

ax = axes[0][0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#9acd32', '#ffbc14']
ax.scatter(delta_X, delta_Y, c=colors)
ax.plot(x, y_pred, color='black', linewidth=0.5)
ax.axvline(x=x_Obs_inp[a], linestyle='--')
ax.axhline(y=x_Obs_out[b], linestyle='--')
ax.set_xlim((x_min-0.5, x_max+0.5))
ax.set_ylim((y_min-0.5, y_max+0.5))
# ax.set_xlabel(r'$\Delta$AMOC 1880-1919 to 1995-2014 [Sv]')
# ax.set_ylabel(r'$\Delta$AMOC 1995-2014 to 2080-2099 [Sv]')
ax.set_xlabel(r'$\Delta$AMOC 1880-1919 to 1975-2014 [Sv]')
ax.set_ylabel(r'$\Delta$AMOC 1975-2014 to 2060-2099 [Sv]')

ax = axes[0][1]
ax.plot(np.exp(log_dens_x_out), x_Obs_out, color='red', lw=1,
            linestyle='-')
ax.hist(Obs_out, density=True, bins=20, color='#add8e6', orientation='horizontal')
ax.axhline(y=x_Obs_out[b], linestyle='--')
ax.set_ylim((y_min-0.5, y_max+0.5))
ax.yaxis.set_label_position("right")
# ax.set_ylabel(r'$\Delta$AMOC-long-term-trend 1995-2014 to 2080-2099 [Sv]')
ax.set_ylabel(r'$\Delta$AMOC-long-term-trend 1975-2014 to 2060-2099 [Sv]')

ax = axes[1][0]
ax.plot(x_Obs_inp, np.exp(log_dens_x_inp), color='red', lw=1,
            linestyle='-')
ax.hist(Obs_inp, density=True, bins=20, color='#add8e6')
ax.axvline(x=x_Obs_inp[a], linestyle='--')
ax.set_xlim((x_min-0.5, x_max+0.5))
# ax.set_xlabel(r'$\Delta$AMOC-long-term-trend 1880-1919 to 1995-2014 [Sv]')
ax.set_xlabel(r'$\Delta$AMOC-long-term-trend 1880-1919 to 1975-2014 [Sv]')

fig.savefig('AMOC_pred_40y.png', bbox_inches='tight')