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
import time



par_vals = {'nstep': 3000, 'dt': 0.01, 'modeltype': 16, 'delta': 1.0, 'lambda': 1.0,'r':0.5,'lambda_std':0.0,
            'Nmax':10000}
# low rate
# par_vals['n']=1
# par_vals['k']=2.0
# par_vals['kw']=3.0  # for low rate k relative to growth required in G1 to bring daughters in line with mothers.

# # high rate
par_vals['n']=1
par_vals['k']=2.0
par_vals['kw']=0.1  # for high rate k relative to growth required in G1 to bring daughters in line with mothers.

# # just right rate
# par_vals = {'nstep': 800, 'dt': 0.01, 'modeltype': 16, 'delta': 1.0, 'lambda': 1.0,'r':0.5,'lambda_std':0.0}
# par_vals['n']=1
# par_vals['k']=2.0
# par_vals['kw']=1.0  # for low rate k relative to growth required in G1 to bring daughters in line with mothers.

temp_vals = []
num_reps = 10
for ind1 in range(num_reps):  # number of replicates
    tic = time.clock()
    init_pop = g.starting_popn(par_vals)
    c, obs,vol_vals = g.discr_time_dilution(par_vals, init_pop)
    temp_vals.append(obs)
    print 'Repeat number {0}, Time taken = {1}'.format(ind1, time.clock()-tic)

tvec = temp_vals[0][1]
std_vec = np.zeros([num_reps, len(tvec)])
av_vec = np.zeros([num_reps, len(tvec)])
cv_vec = np.zeros([num_reps, len(tvec)])
for i1 in range(len(temp_vals)):
        std_vec[i1, :] = temp_vals[i1][6][:]
        av_vec[i1, :] = temp_vals[i1][5][:]
        cv_vec[i1, :] = temp_vals[i1][6][:] / temp_vals[i1][5][:]

#
np.save('./whi5_pert_expt_simulation/noisy_rate_model/dilution_simulations/model16_av_vb_high_rate.npy',av_vec)
np.save('./whi5_pert_expt_simulation/noisy_rate_model/dilution_simulations/model16_std_vb_high_rate.npy',std_vec)
np.save('./whi5_pert_expt_simulation/noisy_rate_model/dilution_simulations/model16_cv_vb_high_rate.npy',cv_vec)