#!/usr/bin/env python

import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import time
from scipy import stats
import pandas as pd
import growth_simulations as g

# This script uses values from the Cerulus et al. paper supplementary, and assesses the results

par1 = {'nstep': 1000, 'dt':0.01, 'td': [1.858, 3.353], 'td_std': [0.4727, 0.9214], 'v_init':1.0, 'r':0.6, 'modeltype':0, 'A_mm':0.129,
    'trans_std_mm':0.9916, 'A_md':0.5086, 'trans_std_md':0.8610}

R2mm = 0.0470
R2md = 0.3254

par1['lambda'] = 1.03089229874*scipy.optimize.fsolve(g.fn, x0=np.log(2.0)/np.mean(par1['td']), args=(par1['td'][0], par1['td'][1]))[0]

init_pop = g.starting_popn(par1)
c,obs = g.discr_time_1(par1, init_pop)

a = [obj.vb for obj in c[10**4:]]
# print np.mean(a)

# print par1['lambda']

fig=plt.figure(figsize=[6,6])
plt.plot(obs[1],np.log(obs[4]), label='data')
vals=scipy.stats.linregress(obs[1][400:], np.log(obs[4][400:]))
plt.plot(obs[1][400:], obs[1][400:]*vals[0]+vals[1], label='linear fit')
plt.xlabel('Time (volume doubling times)')
plt.ylabel('ln(Cell number)')
plt.legend()
plt.title('Cerulus model: population growth rate')
# plt.show()
fig.savefig('./important_plots/growth_rate_cerulus_model.png')
opt_val=scipy.optimize.fsolve(g.fn, x0=np.log(2.0)/np.mean(par1['td']), args=(par1['td'][0], par1['td'][1]))[0]
print 'Relative growth rate = ', vals[0]/opt_val
print 'Error in growth rate = ', (par1['lambda']-vals[0])/par1['lambda']
del fig

fig=plt.figure(figsize=[6,6])
plt.fill_between(obs[1],obs[5]-obs[6], obs[5]+obs[6], label='$\sigma_{V_b}$', alpha=0.3)
plt.plot(obs[1], obs[5], label='data')
# vals=scipy.stats.linregress(obs[1][400:], np.log(obs[-1][400:]))
# plt.plot(obs[1][400:], obs[1][400:]*vals[0]+vals[1], label='linear fit')
plt.xlabel('Time (volume doubling times)')
plt.ylabel(r'$\langle V_b\rangle$')
plt.legend()
plt.title('Cerulus model: Cell volume over time')
# plt.show()
fig.savefig('./important_plots/av_vol_cerulus_model.png')
del fig

fig=plt.figure(figsize=[6,6])
plt.plot(obs[1],obs[6]/obs[5],label='CV')
# plt.plot(obs[1], obs[5])
# vals=scipy.stats.linregress(obs[1][400:], np.log(obs[-1][400:]))
# plt.plot(obs[1][400:], obs[1][400:]*vals[0]+vals[1], label='linear fit')
plt.xlabel('Time (volume doubling times)')
plt.ylabel('CV($V_b$)')
plt.legend()
plt.title('Cerulus model: CV in $V_b$ over time')
# plt.show()
fig.savefig('./important_plots/CV_vol_cerulus_model.png')
del fig

vb=[]
tgrow = []
print 'r', par1['r'], 'modeltype', par1['modeltype']
for i0 in range(2):
    vb.append([obj.vb for obj in c[10**4:] if obj.celltype==i0])
    tgrow.append([obj.t_grow for obj in c[10**4:] if obj.celltype == i0])
    print 'celltype', i0
    print '<V_b>', np.mean(vb[-1]),'sigma_{V_b}', np.std(vb[-1]), 'CV(Vb)',scipy.stats.variation(vb[-1])
    print '<t_d>', np.mean(tgrow[-1]), 'sigma_{t_d}', np.std(tgrow[-1])


t1 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))
t2 = scipy.stats.zscore(np.asarray([obj.parent.t_grow for obj in c[10**4:-1] if obj.celltype == 0]))

t3 = scipy.stats.zscore(np.asarray([obj.t_grow for obj in c[10**4:-1] if obj.celltype == 1]))
t4 = scipy.stats.zscore(np.asarray([obj.parent_current.t_grow for obj in c[10**4:-1] if obj.celltype == 1]))

# print len(t1), len(t2), len(t3), len(t4)
vals = []
vals.append(scipy.stats.linregress(t2, t1))
vals.append(scipy.stats.linregress(t4, t3))
print 'R^2 for mother-mother correlations', vals[0][2]**2
print 'R^2 for mother-daughter correlations', vals[1][2]**2

fig = plt.figure(figsize=[6,6])
plt.hexbin(t2, t1)
temp = scipy.stats.linregress(t2,t1)
xvals = np.linspace(np.mean(t2)-2.0*np.std(t2), np.mean(t2)+2.0*np.std(t2), 100)
plt.plot(xvals, temp[0]*xvals+temp[1], label='$R^2$={0}'.format(np.round(temp[2]**2,4)))
plt.xlabel('Z score Mothers 1')
plt.ylabel('Z score Mothers 2')
plt.title('Mother - mother correlation')
plt.legend()
fig.savefig('./important_plots/mother_mother_cerulus_model.png')
del fig

fig = plt.figure(figsize=[6,6])
plt.hexbin(t4, t3)
temp = scipy.stats.linregress(t4,t3)
xvals = np.linspace(np.mean(t4)-2.0*np.std(t4), np.mean(t4)+2.0*np.std(t4), 100)
plt.plot(xvals, temp[0]*xvals+temp[1], label='$R^2$={0}'.format(np.round(temp[2]**2, 4)))
plt.xlabel('Z score Mothers')
plt.ylabel('Z score Daughters')
plt.title('Daughter - mother correlation')
plt.legend()
fig.savefig('./important_plots/daughter_mother_cerulus_model.png')
del fig