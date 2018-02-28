#!/usr/bin/env python

import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import time
from scipy import stats
import pandas as pd
import growth_simulations as g

td = [0]
td_std = [0.1, 0.1]
# td = [1.0, 1.0]
# td_std = [0.1, 0.1]

# this script uses a simple adder model

par1 = {'nstep': 1000, 'dt':0.01, 'td_std': td_std, 'v_init':1.0, 'modeltype':1, 'delta':1.0, 'num_gen1':10}

# defining r and lambda to approximate the same average division times for daughters and mothers.

par1['lambda']=np.log(2)
par1['r']=1.0

init_pop = g.starting_popn(par1)
# c,obs = g.discr_time_1(par1, init_pop)
c = g.discr_gen_1(par1, init_pop)

print len(c)
temp1 = [obj for obj in c[10**4:] if obj.celltype == 1]
tgrowth = [[],[]]
for obj in temp1:
    tgrowth[1].append(obj.t_grow)
    tgrowth[0].append(obj.parent_current.t_grow)
# temp = [obj]
temp2 = scipy.stats.linregress(scipy.stats.zscore(np.asarray(tgrowth[0])),scipy.stats.zscore(np.asarray(tgrowth[1])))
print 'mother daughter correlation $R^2$', temp2[2]**2

temp = [obj for obj in c[10**4:] if obj.celltype == 0]
tgrowth = [[],[]]
for obj in temp:
    tgrowth[1].append(obj.t_grow)
    tgrowth[0].append(obj.parent.t_grow)
# temp = [obj]
temp2 = scipy.stats.linregress(scipy.stats.zscore(np.asarray(tgrowth[0])),scipy.stats.zscore(np.asarray(tgrowth[1])))
print 'mother mother correlation', temp2[2]**2

# print 100.0*(-len(temp)+len(temp1))/len(temp)