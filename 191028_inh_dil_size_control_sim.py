import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import growth_simulations as g
from matplotlib import rc
rc('text', usetex=True)
font = {'weight' : 'bold',
        'size'   : 22}
plt.rc('font', **font)
import seaborn as sns

par_vals = {'nstep': 800, 'dt': 0.01, 'modeltype': 15, 'delta': 1.0, 'lambda': 1.0,'r':0.5,'lambda_std':0.0}
par_vals['n']=1
par_vals['k']=2.0
temp_vals = []
for ind1 in range(10):  # number of replicates
    init_pop = g.starting_popn(par_vals)
    c, obs,vol_vals = g.discr_time_1(par_vals, init_pop)
    temp1=[]
    temp1.append(obs)
    for ind in range(5):
        init_pop1 = g.starting_popn_seeded(c,par_vals)
        c, obs,vol_vals = g.discr_time_1(par_vals, init_pop1)
        temp1.append(obs)
        print 'Done Repeat {0}, iteration {1}'.format(ind1,ind)
    temp_vals.append(temp1)

tvec = np.linspace(0.0, (par_vals['nstep'] + 1) * len(temp1) * par_vals['dt'] * np.log(2) / par_vals['lambda'],
                   (par_vals['nstep'] + 1) * len(temp1))
# print tvec
std_vec = np.zeros([10, len(tvec)])
av_vec = np.zeros([10, len(tvec)])
cv_vec = np.zeros([10, len(tvec)])
for i1 in range(len(temp_vals)):
    for i0 in range(len(temp_vals[0])):
        std_vec[i1, i0 * (par_vals['nstep'] + 1):(i0 + 1) * (par_vals['nstep'] + 1)] = temp_vals[i1][i0][6][:]
        av_vec[i1, i0 * (par_vals['nstep'] + 1):(i0 + 1) * (par_vals['nstep'] + 1)] = temp_vals[i1][i0][5][:]
        cv_vec[i1, i0 * (par_vals['nstep'] + 1):(i0 + 1) * (par_vals['nstep'] + 1)] = temp_vals[i1][i0][6][:] / \
                                                                                      temp1[i0][5][:]

np.save('./whi5_pert_expt_simulation/noisy_rate_model/model15_av_vb1.npy',av_vec)
np.save('./whi5_pert_expt_simulation/noisy_rate_model/model15_std_vb1.npy',std_vec)
np.save('./whi5_pert_expt_simulation/noisy_rate_model/model15_cv_vb1.npy',cv_vec)