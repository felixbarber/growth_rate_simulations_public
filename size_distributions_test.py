import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import time
from scipy import stats
import pandas as pd
import growth_simulations as g

par1 = {'nstep': 500, 'dt':0.01, 'td': [1.858, 3.353], 'td_std': [0.4727, 0.9214], 'v_init':1.0, 'r':0.6, 'modeltype':0, 'A_mm':0.129,
    'trans_std_mm':0.9916, 'A_md':0.5086, 'trans_std_md':0.8610}

R2mm = 0.0470
R2md = 0.3254

par1['lambda'] = scipy.optimize.fsolve(g.fn, x0=np.log(2.0)/np.mean(par1['td']), args=(par1['td'][0], par1['td'][1]))[0]

init_pop = g.starting_popn(par1)
c,obs = g.discr_time_1(par1, init_pop)

a = [obj.vb for obj in c if obj.exists]
print np.mean(a)

# Fig S2 in Cerulus et al. shows how they extract values A and trans_std from the correlation between mother and daughter
# cells.

# data values taken from FIG6_WT strain BY4741 in 2% palatinose. See dataset S4
