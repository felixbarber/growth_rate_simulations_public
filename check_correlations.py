#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import growth_simulation_dilution_asymmetric as g
import time
import scipy
from scipy import stats
import os.path


font = {'family': 'normal', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

# as a first pass we will just make cells not shrink, have small noise in Whi5 production, in passage through Start,
# and in timing of G2 (CD parameter below). We will, however, include noise in cell volume growth rate. This will be
# a noisy integrator for Whi5 production

delta = 10.0

par_vals = dict([('dt', 0.01), ('td', 1.0), ('num_s1', 500), ('nstep', 500), ('num_gen', 10), ('modeltype', 17),
                 ('delta', 10.0), ('d_std', 0.02), ('g1_thresh_std', 0.02), ('g1_std', 0.0), ('g1_delay', 0.0),
                ('l_std', 0.2), ('CD', 0.6), ('g2_std', 0.2)])

celltype = ['Mothers', 'Daughters']
path = './check_correlations_output'

full_title = ['Mothers $t_d$ z-score', 'Daughters $t_d$ z-score']
sub_title = [r'Growth rate $\sigma={0}$'.format(par_vals['l_std'])]
# while not os.path.isfile(path+'_td_celltype+{0}'.format(0)+'.npy'):  # ensures that this dataset is there
temp = g.discr_gen(par_vals)
temp1 = g.starting_popn_seeded([obj for obj in temp if obj.exists], par_vals)
temp2 = g.discr_time_1(par_vals, temp1)
td = []
td.append(np.asarray([max(obj.t_grow,0.0) for obj in temp2[0] if obj.exists and obj.isdaughter == 1]))
print len(td[-1])
td.append(np.asarray([max(obj.mother.nextgen.t_grow,0.0) for obj in temp2[0] if obj.exists and obj.isdaughter == 1]))
print len(td[-1])
nan_inds = ~(np.isnan(td[0])+np.isnan(td[1]))
for i1 in range(2):
    x = [obj.vb for obj in temp2[0][1000:] if obj.isdaughter == i1]
    y = [obj.vd for obj in temp2[0][1000:] if obj.isdaughter == i1]

    temp3 = scipy.stats.linregress(x, y)
    print celltype[i1]+' linear regression ', temp3[0]
    np.save(path+'_td_celltype+{0}'.format(i1), td[i1])
temp4 = scipy.stats.pearsonr(td[0][nan_inds], td[1][nan_inds])
print 'mother daughter doubling time PCC', temp4[0]
temp5 = [np.zeros([np.sum(nan_inds), 2])]
temp_val1 = td[1][nan_inds]  # mothers
temp_val2 = td[0][nan_inds]   # daughters
temp5[0][:, 0] = scipy.stats.mstats.zscore(temp_val1)
temp5[0][:,1] = scipy.stats.mstats.zscore(temp_val2)   # daughters

fig=g.plot_vals_array(temp5, sub_title, full_title)
fig.savefig('./growth_rate_project/check_correlations_td.png')
del temp, temp1, temp2, temp3, temp4, temp5


