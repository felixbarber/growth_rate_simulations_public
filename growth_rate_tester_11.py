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

# this script uses a simple adder model and takes r=0.5


par1 = {'nstep': 1000, 'dt':0.01, 'td': None, 'td_std': None, 'v_init':1.0, 'modeltype':1, 'delta':1.0, 'lambda_std':None}

# defining r and lambda simply

par1['lambda'] = 1.0
par1['r'] = 0.5

# parameters to store: number growth rate, volume growth rate, mean interdivision time for mothers and daughters, PCC
# and R values for interdivision time correlations.

# At this stage all we vary is the noise in interdivision times.
vals, offset, gr_vals = np.linspace(0.0, 0.2, 9), 0.001, np.linspace(0.0, 0.2, 9)
num_outputs = 10  # parameters to store: number growth rate, volume growth rate, mean interdivision time for mothers and daughters, PCC
# and R values for interdivision time correlations.
output_data = np.zeros([num_outputs, len(vals), len(vals), len(gr_vals)])

for i0 in range(len(vals)):
    for i1 in range(len(vals)):
        for i2 in range(len(gr_vals)):
            # print i0, i1, i2
            td_std = [vals[i0]+offset, vals[i1]+offset]  # mother, daughter
            par1['td_std'] = td_std
            if i2 == 0:
                par1['lambda_std'] = None
            else:
                par1['lambda_std'] = gr_vals[i2]+offset

            init_pop = g.starting_popn(par1)
            c, obs = g.discr_time_1(par1, init_pop)

            temp1 = scipy.stats.linregress(obs[1][400:], np.log(obs[4][400:]))
            output_data[0, i0, i1, i2] = temp1[0]  # population number growth rate
            temp1 = scipy.stats.linregress(obs[1][400:], np.log(obs[7][400:]))
            output_data[1, i0, i1, i2] = temp1[0]  # population volume growth rate
            del temp1

            for temp_ind in range(2):
                tgrow = [obj.t_grow for obj in c[10**4:] if obj.celltype == temp_ind]
                output_data[2+temp_ind, i0, i1, i2] = np.mean(tgrow)

            t1 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))
            t2 = scipy.stats.zscore(np.asarray([obj.parent.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))
            temp2 = scipy.stats.linregress(t2, t1)
            output_data[4, i0, i1, i2] = temp2[2]  # mother mother R value
            temp2 = scipy.stats.pearsonr(t1, t2)
            output_data[5, i0, i1, i2] = temp2[0]  # mother mother PCC

            t1 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10 ** 4:-1] if obj.celltype == 1]))
            t2 = scipy.stats.zscore(np.asarray([obj.parent.t_grow for obj in c[10 ** 4:-1] if obj.celltype == 1]))
            temp2 = scipy.stats.linregress(t2, t1)
            output_data[6, i0, i1, i2] = temp2[2]  # mother daughter R value
            temp2 = scipy.stats.pearsonr(t1, t2)
            output_data[7, i0, i1, i2] = temp2[0]  # mother daughter PCC

            t1 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10 ** 4:-1] if obj.celltype == 1]))
            t2 = scipy.stats.zscore(np.asarray([obj.parent_current.t_grow for obj in c[10 ** 4:-1] if obj.celltype == 1]))
            temp2 = scipy.stats.linregress(t2, t1)
            output_data[8, i0, i1, i2] = temp2[2]  # mother daughter R value same generation
            temp2 = scipy.stats.pearsonr(t1, t2)
            output_data[9, i0, i1, i2] = temp2[0]  # mother daughter PCC same generation

            stored_vals = np.zeros([len(obs), len(obs[1])])
            for i4 in range(len(obs)):
                stored_vals[i4, :] = obs[i4][:]
            np.save('./data/growth_rate_tester_11_{0}_{1}_{2}.npy'.format(i0, i1, i2), stored_vals)
            print 'finished', i0, i1, i2
            del temp1, tgrow, temp2, t1, t2, init_pop, c, obs, stored_vals
np.save('./growth_rate_tester_11_output_data',output_data)
