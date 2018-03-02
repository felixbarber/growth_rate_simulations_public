#!/usr/bin/env python

import numpy as np
import growth_simulations as g

vals = np.load('./growth_rate_tester_5_output_data.npy')
labels = [['Measured growth rate', 'Inferred growth rate', 'Mother-mother $R^2$','Mother-daughter $R^2$'],
          ['Daughters $\sigma_{t_d}$ [$t_d$]', 'Mothers $\sigma_{t_d}$ [$t_d$]']]
m_std, d_std = np.linspace(0.0,1.0,11), np.linspace(0.0,1.0,11)
figs = g.heat_maps(vals, labels, x_std=d_std, y_std=m_std)

for i0 in range(len(figs)):
    figs[i0].savefig('./important_plots/adder_obs_{0}_noiseless_lambda.png'.format(i0))
    print vals[i0,0,0], vals[i0,5,0]

vals = np.load('./growth_rate_tester_7_output_data.npy')
labels = [['Measured growth rate $\sigma_\lambda/\lambda=0.2$', 'Inferred growth rate $\sigma_\lambda/\lambda=0.3$',
           'Mother-mother $R^2$ $\sigma_\lambda/\lambda=0.3$','Mother-daughter $R^2$ $\sigma_\lambda/\lambda=0.3$'],
          ['Daughters $\sigma_{t_d}$ [$t_d$]', 'Mothers $\sigma_{t_d}$ [$t_d$]']]
m_std, d_std = np.linspace(0.0,1.0,11), np.linspace(0.0,1.0,11)
figs = g.heat_maps(vals, labels, x_std=d_std, y_std=m_std)

for i0 in range(len(figs)):
    figs[i0].savefig('./important_plots/adder_obs_{0}_noisy_lambda.png'.format(i0))
    print vals[i0,0,0], vals[i0,5,0]