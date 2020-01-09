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

td = [1.858, 3.353]
td_std = [0.4727, 0.9214]
# td = [1.0, 1.0]
# td_std = [0.1, 0.1]

# this script uses a simple adder model


par1 = {'nstep': 1000, 'dt':0.01, 'td': td, 'td_std': td_std, 'v_init':1.0, 'modeltype':1, 'A_mm':0.129,
    'trans_std_mm':0.9916, 'A_md':0.5086, 'trans_std_md':0.8610, 'delta':1.0, 'lambda_std':0.2}

# defining r and lambda to approximate the same average division times for daughters and mothers.

par1['lambda'] = scipy.optimize.fsolve(g.fn_1, x0=np.log(2.0) / np.mean(td), args=(par1['td'][0], par1['td'][1]))[0]
temp = 0.5/(np.exp(par1['lambda']*par1['td'][1])-1)
par1['r'] = temp/(1-temp)

# At this stage all we vary is the noise in interdivision times.
vals, offset, gr_vals, vals1 =  np.linspace(0.0,1.0,11), 0.01, np.linspace(0.0, 0.2, 9), np.linspace(0.0, 2.0, 11)
num_outputs = 6  # growth rate, optimal growth rate, R2 mm, R2 md, R2 md diff gen, volume growth rate
output_data = np.zeros([num_outputs, 11, 11, len(gr_vals)])
for i0 in range(11):
    for i1 in range(11):
        for i2 in range(len(gr_vals)):
            print i0, i1, i2
            td_std = [vals[i0]+offset, vals1[i1]+offset]
            par1['td_std'] = td_std
            par1['lambda_std'] = gr_vals[i2]+offset
            init_pop = g.starting_popn(par1)
            c,obs = g.discr_time_1(par1, init_pop)

            temp1 = scipy.stats.linregress(obs[1][400:], np.log(obs[4][400:]))
            tgrow = []
            for i3 in range(2):
                tgrow.append([obj.t_grow for obj in c[10**4:] if obj.celltype == i3])
            opt_val=scipy.optimize.fsolve(g.fn, x0=np.log(2.0)/np.mean(par1['td']), args=(np.mean(tgrow[0]), np.mean(tgrow[1])))[0]
            output_data[0, i0, i1, i2] = temp1[0]
            output_data[1, i0, i1, i2] = opt_val

            t1 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))
            t2 = scipy.stats.zscore(np.asarray([obj.parent.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))

            t3 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10**4:-1] if obj.celltype == 1]))
            t4 = scipy.stats.zscore(np.asarray([obj.parent_current.t_grow for obj in c[10**4:-1] if obj.celltype == 1]))

            t5 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10 ** 4:-1] if obj.celltype == 1]))
            t6 = scipy.stats.zscore(
                np.asarray([obj.parent.t_grow for obj in c[10 ** 4:-1] if obj.celltype == 1]))

            temp2 = []
            temp2.append(scipy.stats.linregress(t2, t1))
            temp2.append(scipy.stats.linregress(t4, t3))
            temp2.append(scipy.stats.linregress(t6, t5))
            output_data[2, i0, i1, i2] = temp2[0][2]**2
            output_data[3, i0, i1, i2] = temp2[1][2]**2
            output_data[4, i0, i1, i2] = temp2[2][2] ** 2
            temp3 = scipy.stats.linregress(obs[1][400:], np.log(obs[7][400:]))
            output_data[5, i0, i1, i2] = temp3[0]

            stored_vals = np.zeros([len(obs), len(obs[1])])
            for i4 in range(len(obs)):
                stored_vals[i4, :] = obs[i4][:]
            np.save('./data/growth_rate_tester_8_{0}_{1}_{2}.npy'.format(i0, i1, i2), stored_vals)
            print 'finished', i0, i1, i2
            del temp1, tgrow, temp2, t1, t2, t3, t4, opt_val, init_pop, c, obs, stored_vals
np.save('./growth_rate_tester_8_output_data',output_data)
