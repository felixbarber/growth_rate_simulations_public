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