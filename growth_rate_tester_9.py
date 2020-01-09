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

vals = np.load('./growth_rate_tester_8_output_data.npy')
num_outputs = 6  # growth rate, optimal growth rate, R2 mm, R2 md, R2 md diff gen, volume growth rate


labels = [['Measured growth rate', 'Inferred growth rate', 'Mother-mother $R^2$','Mother-daughter same generation $R^2$',
           'Mother-daughter different generation $R^2$', 'Volume growth rate' ],
          ['Daughters $\sigma_{t_d}$ [$t_d$]', 'Mothers $\sigma_{t_d}$ [$t_d$]']]
offset = 0.01
m_std, d_std, l_std = np.linspace(0.0, 1.0, 11)+offset, np.linspace(0.0,1.0,11)+offset, np.linspace(0.0, 0.2, 9)+offset
temp_labels = []
temp_ind = 0  # do this for l_std=0
temp_labels.append([temp+' $\sigma_\lambda/\lambda=${0}'.format(np.round(l_std[temp_ind], 3)) for temp in labels[0]])
temp_labels.append(labels[1])
print temp_labels
figs = g.heat_maps(np.squeeze(vals[:,:,:,temp_ind]), temp_labels, x_std=d_std, y_std=m_std)

for i0 in range(len(figs)):
    figs[i0].savefig('./growth_rate_tester_8_plots/adder_{0}_noiseless_lambda.png'.format(i0))
    print vals[i0,0,0, 0], vals[i0,5,0, 0]

temp_labels = []
temp_ind = 8  # do this for l_std=0
temp_labels.append([temp + ' $\sigma_\lambda/\lambda=${0}'.format(np.round(l_std[temp_ind], 3)) for temp in labels[0]])
temp_labels.append(labels[1])
print temp_labels
figs = g.heat_maps(np.squeeze(vals[:, :, :, temp_ind]), temp_labels, x_std=d_std, y_std=m_std)

for i0 in range(len(figs)):
    figs[i0].savefig('./growth_rate_tester_8_plots/adder_{0}_noisy_lambda.png'.format(i0))
    print vals[i0, 0, 0, 0], vals[i0, 5, 0, 0]
#
# vals = np.load('./growth_rate_tester_7_output_data.npy')
# labels = [['Measured growth rate $\sigma_\lambda/\lambda=0.2$', 'Inferred growth rate $\sigma_\lambda/\lambda=0.3$',
#            'Mother-mother $R^2$ $\sigma_\lambda/\lambda=0.3$','Mother-daughter $R^2$ $\sigma_\lambda/\lambda=0.3$'],
#           ['Daughters $\sigma_{t_d}$ [$t_d$]', 'Mothers $\sigma_{t_d}$ [$t_d$]']]
# m_std, d_std = np.linspace(0.0,1.0,11), np.linspace(0.0,1.0,11)
# figs = g.heat_maps(vals, labels, x_std=d_std, y_std=m_std)
#
# for i0 in range(len(figs)):
#     figs[i0].savefig('./important_plots/adder_obs_{0}_noisy_lambda.png'.format(i0))
#     print vals[i0,0,0], vals[i0,5,0]