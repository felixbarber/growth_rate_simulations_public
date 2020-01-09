#!/usr/bin/env python

import numpy as np
import scipy
import scipy.optimize
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import time
from scipy import stats
import pandas as pd
import growth_simulations as g

font = {'family': 'normal', 'weight': 'bold', 'size': 22}
plt.rc('font', **font)

vals = np.load('./growth_rate_tester_11v2_output_data.npy')
num_outputs = 6  # growth rate, optimal growth rate, R2 mm, R2 md, R2 md diff gen, volume growth rate
r2=True
if r2:
    r_list = [4, 6, 8]
    for ind in r_list:
        vals[ind,:,:,:] = np.absolute(vals[ind,:,:,:])*vals[ind,:,:,:]

labels = [['Number growth rate', 'Volume growth rate', 'Mother <t grow>','Daughter <t grow>', 'Mother-mother $R^2$',
           'Mother-mother PCC', 'Mother-daughter different generation $R^2$', 'Mother-daughter different generation PCC',
           'Mother-daughter same generation $R^2$', 'Mother-daughter same generation PCC'],
          ['Daughters $\sigma_{t_d}$ [$t_d$]', 'Mothers $\sigma_{t_d}$ [$t_d$]', '$\sigma_{\lambda}$']]
offset = 0.001
m_std, d_std, l_std = np.linspace(0.0, 0.2, 9)+offset, np.linspace(0.0, 0.2, 9)+offset, np.linspace(0.0, 0.2, 9)
temp_labels = []
temp_ind = 0  # do this for l_std=0
temp_labels.append([temp+', $\sigma_\lambda/\lambda=${0}'.format(np.round(l_std[temp_ind], 3)) for temp in labels[0]])
temp_labels.append(labels[1])
print temp_labels
figs = g.heat_maps(np.squeeze(vals[:, :, :, temp_ind]), temp_labels, x_std=d_std, y_std=m_std)

for i0 in range(len(figs)):
    figs[i0].savefig('./growth_rate_tester_11_plots/adder_{0}_lambda_{1}.png'.format(i0, temp_ind))
    print vals[i0, 0, 0, 0], vals[i0, 5, 0, 0]

temp_labels = []
temp_ind = 8  # do this for l_std=0.2
temp_labels.append([temp + ', $\sigma_\lambda/\lambda=${0}'.format(np.round(l_std[temp_ind], 3)) for temp in labels[0]])
temp_labels.append(labels[1])
print temp_labels
figs = g.heat_maps(np.squeeze(vals[:, :, :, temp_ind]), temp_labels, x_std=d_std, y_std=m_std)

for i0 in range(len(figs)):
    figs[i0].savefig('./growth_rate_tester_11_plots/adder_{0}_lambda_{1}.png'.format(i0, temp_ind))
    print vals[i0, 0, 0, 0], vals[i0, 5, 0, 0]

temp_labels = []
temp_ind = 4  # do this for l_std=0.1
temp_labels.append([temp + ', $\sigma_\lambda/\lambda=${0}'.format(np.round(l_std[temp_ind], 3)) for temp in labels[0]])
temp_labels.append(labels[1])
print temp_labels
figs = g.heat_maps(np.squeeze(vals[:, :, :, temp_ind]), temp_labels, x_std=d_std, y_std=m_std)

for i0 in range(len(figs)):
    figs[i0].savefig('./growth_rate_tester_11_plots/adder_{0}_lambda_{1}.png'.format(i0, temp_ind))
    print vals[i0, 0, 0, 0], vals[i0, 5, 0, 0]