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
    'trans_std_mm':0.9916, 'A_md':0.5086, 'trans_std_md':0.8610, 'delta':1.0}

# defining r and lambda to approximate the same average division times for daughters and mothers.

par1['lambda'] = scipy.optimize.fsolve(g.fn_1, x0=np.log(2.0) / np.mean(td), args=(par1['td'][0], par1['td'][1]))[0]
temp = 0.5/(np.exp(par1['lambda']*par1['td'][1])-1)
par1['r'] = temp/(1-temp)

# At this stage all we vary is the noise in interdivision times.
vals, offset =  np.linspace(0.0,1.0,11), 0.01
num_outputs = 4  # growth rate, optimal growth rate, R2 mm, R2 md
output_data = np.zeros([4, 11, 11])
for i0 in range(11):
    for i1 in range(11):
        td_std = [vals[i0]+offset, vals[i1]+offset]
        par1['td_std'] = td_std
        init_pop = g.starting_popn(par1)
        c,obs = g.discr_time_1(par1, init_pop)
        temp1 = scipy.stats.linregress(obs[1][400:], np.log(obs[4][400:]))
        tgrow = []
        for i2 in range(2):
            tgrow.append([obj.t_grow for obj in c[10**4:] if obj.celltype == i2])
        opt_val=scipy.optimize.fsolve(g.fn, x0=np.log(2.0)/np.mean(par1['td']), args=(np.mean(tgrow[0]), np.mean(tgrow[1])))[0]
        output_data[0, i0, i1] = temp1[0]
        output_data[1, i0, i1] = opt_val

        t1 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))
        t2 = scipy.stats.zscore(np.asarray([obj.parent.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))

        t3 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10**4:-1] if obj.celltype == 1]))
        t4 = scipy.stats.zscore(np.asarray([obj.parent_current.t_grow for obj in c[10**4:-1] if obj.celltype == 1]))

        temp2 = []
        temp2.append(scipy.stats.linregress(t2, t1))
        temp2.append(scipy.stats.linregress(t4, t3))
        output_data[2, i0, i1] = temp2[0][2]**2
        output_data[3, i0, i1] = temp2[1][2]**2
        print 'finished', i0, i1
        del temp1, tgrow, temp2, t1, t2, t3, t4, opt_val, init_pop, c, obs
np.save('./growth_rate_tester_5_output_data',output_data)
