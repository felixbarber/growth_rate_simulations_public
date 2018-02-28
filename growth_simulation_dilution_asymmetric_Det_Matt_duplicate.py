import numpy as np
import scipy
import weakref
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import stats
import matplotlib.colors as colors
import matplotlib.cm as cmx
from shapely.geometry import LineString
from descartes import PolygonPatch
import pandas as pd
from skimage import measure
sns.set(font_scale=2)
plt.rc('text', usetex=True)

def test(x, y, z):
    return np.array([x+y+z**3])

# font = {'family': 'normal', 'weight': 'bold', 'size': 12}
# plt.rc('font', **font)

########################################################################################################################
# DISCRETIZED TIME SIMULATIONS
########################################################################################################################

# This notebook has asymmetric division for yeast.


# This defines a global variable within growth_simulation which is necessary to define the Mother and Daughter classes.
# Anything that is defined in this variable is fixed for the course of the simulation. If you wish to iterate over a v
# variable you should include that as an input parameter for the function discr_time1

par = dict([('num_s', 50), ('vd', 1.0), ('vm', 1.0), ('wd', 1.0), ('wm', 1.0), ('std_v', 0.2), ('std_w', 0.2)])
# w_d = 2 * par1['K'] * par1['CD'] * \
#         (1 - np.exp(0.5 * (np.log(2) * par1['g2_std'] / par1['td']) ** 2) / 2 ** (par1['CD'] / par1['td']))
#     w_m = 2 * par1['K'] * par1['CD'] * np.exp(0.5*(np.log(2)*par1['g2_std']/par1['td'])**2)/2**(par1['CD']/par1['td'])

# This notebook requires par1 for the simulations to run. The inputs for par1 should include the following in
# the following units:

# 'td' = 1.0  # doubling time. Varying this in theory shouldn't change anything, since all other variables
# are defined relative to the doubling time.
# ['CD']    # CD period as fraction of td -- sims
# ['K']     # Whi5 synthesis rate in volume per doubling time -- sims
# g1_std    # Noise in Whi5 Start threshold in units of volume -- sims
# g2_std    # unit less fraction of td, NOT of cd. -- sims
# ['l_std'] # unit less fraction of growth rate -- sims
# ['k_std'] # unit less fraction of synthesis rate K -- sims

models = ['discr gen noiseless whi5 adder 0', 'discr gen noisy whi5 adder 1', 'volumetric distr noise noiseless whi5 adder 2',
          'volumetric distr noise noisy whi5 adder 3', 'Constant whi5 adder 4',
          'Noisy synthesis rate adder 5', '6', '7', '8', 'Constant Whi5 adder with no neg growth 9',
          'Noisy Whi5 adder with no neg growth 10', 'Constant Whi5 adder with const frac of division and no neg growth 11',
          'Constant Whi5 adder with const frac of division and neg growth 12', 'Const Whi5 adder and fraction with neg growth 13',
          'Const Whi5 adder and fraction with no neg growth 14', 'Noisy Whi5 adder and const fraction with neg growth 15',
          'Noisy Whi5 adder and const fraction with no neg growth 16', 'Noisy Integrating adder 17',
          'Noisy Integrating adder with no neg growth 18', 'Noisy integrating adder with timer mothers 19',
          'Noisy integrating adder with timer mothers and no neg growth 20', 'Noisy Integrating adder with neg growth and const Whi5 frac 21',
          'Noisy Integrating adder with no neg growth and const Whi5 frac 22', 'Noise in r integrating adder 23',
          'Noise in r integrating adder no neg growth 24']

# 8 = symmetric division noiseless adder
list1 = [4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # models which include noise in the growth rate
list2 = [1, 3, 5, 10, 15, 16]  # models which have noise in the production of Whi5 due to noise in G2 timing
list2a = [6]  # models which have gaussian noise in production of Whi5 uncorrelated with noise in timing
list2b = [5, 10, 15, 16]  # models with noise in Whi5 synthesis rate
list3 = [0, 2, 4, 7, 8, 9, 11, 12, 13, 14]  # models which have a noiseless Whi5 adder during G2.
list3a = [17, 18, 19, 20, 21, 22, 23, 24]  # models which have a noisy integrator model for Whi5 accumulation
list4 = [2, 3]  # models which have a constant asymmetry in division (asymmetry defined here by 1+r=exp[l*CD])
list5 = [0, 1, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # models which have all noise in budded
# phase go to bud (i.e. noise in division asymmetry)
list6 = [7]  # models with multiplicative noise in passage through Start
list7 = [8, 11, 12]  # models with a constant fraction given to daughters of par1['frac']
list8 = [9, 10, 11, 14, 16, 18, 20, 22, 24, 26]  # models which have a minimum volume (vb) at which they go through Start'
list9 = [13, 14, 15, 16, 21, 22]  # models with a constant non-volumetric Whi5 fraction given to daughters and mothers
list10 = [19, 20]  # models with different roles for mothers and daughters
list11 = [23, 24, 25, 26]  # models which have noise in r. Assumes no noise in growth rate or r.
list12 = [25, 26]  # model in which noise in r is translated to noise in t
list13 = [27]  # model in which cells can't shrink, and just act as adders between birth and division


# par['CD'] = 0.75*par['td']
# par['k'] = 2**(1-par['CD']/par['td'])/par['CD']


# dt=time step
# nstep=number of time steps we go through
# td is the population doubling time
# num_s is the initial number of mothers and daughters
# Vo is the mean of the initial volume distribution.
# std_iv is the standard deviation of the starting volume distribution (same for both mothers and daughters)
# std_iw is the standard deviation of the starting whi5 distribution (same for both mothers and daughters)
# g1_std is the standard deviation in the passage time through g1
# g2_std is the standard deviation in the passage time through S/G2/M
# r is the asymmetry ratio in volume at division for mother and daughter cells


class Cell(object):
    # 'common base class for all mother cells of different growth policies'
    cellCount = 0  # total number of cells
    negcellCount = 0  # number of cells with a cell division time sufficiently low relative to their time of birth that
    # they are neglected.
    b = 1.0

    def __init__(self, vb, w, tb, mother=None):
        # produces a new instance of the mother class. Only instance variables
        # initially are the initiator abundance at birth (w), the volume at birth,
        # the time of birth and it points to the mother.
        self.vb = vb
        self.wb = w
        self.tb = tb
        self.mother = mother  # references the mother cell
        self.daughter = None  # weakly references the daughter cell
        self.nextgen = None  # weakly references the cell this gives rise to after the next division event.
        self.isdaughter = False  # boolean for whether the cell is a mother or daughter
        self.exists = True  # indexes whether the cell exists currently. Use in combination with should_div to check
        # whether some small cells are managing to skip division.
        self.should_divide = False  # indexes whether a cell should have divided during the simulation (quality control)
        self.gen_m = 0
        self.gen_d = 0

    def grow(self, par1):  # integration model for accumulation of initiator protein. In slow growth case.
        self.td = par1['td']  # mass doubling time constant. All models get this

        if par1['modeltype'] in list13:
            l = (1 + np.random.normal(0.0, par1['l_std'], 1)[0]) * np.log(2) / self.td  # noisy growth rate
            d = (1 + np.random.normal(0.0, par1['d_std'], 1)[0]) * par1['delta']
            self.vd = self.vb+d  # no noise in this
            self.vi = max(self.vb, self.vd/(1+par1['r']))
            # placeholders
            self.wd = 0
        else:
            if not par1['modeltype'] in list11:  # if these cells don't have noise entirely in the asymmetry ratio
                self.CD = par1['CD']  # time delay C+D after the initiation is begun
            if par1['modeltype'] in list3 + list2a + list2b:  # in this case we have a model with delta =K*CD
                self.K = par1['K'] / par1['td']  # Whi5 synthesis rate per second
                self.delta = par1['K']*par1['CD']
            if par1['modeltype'] in list3a:
                self.delta = par1['delta']  # in this case delta is defined separately as some integrated value delta.
            # Calculate the volume of this cell at initiation and division.
            if par1['modeltype'] in list12:
                self.delta = par1['K']*(np.log(1+par1['r'])-0.5*par1['r_std']**2/(1+par1['r'])**2)/np.log(2)
                self.K = par1['K']

            # at this stage all cells should have defined Delta

            if par1['g1_thresh_std'] != 0:  # gaussian noise in the abundance of initiator required to cause initiation
                noise_thr = np.random.normal(0.0, par1['g1_thresh_std'], 1)[0]
            else:
                noise_thr = 0.0
            if par1['modeltype'] in list3a+list2b+list12:
                noise_thr = noise_thr * self.delta  # scaling of noise in G1 by Delta.
            if par1['g1_std'] != 0:
                noise_g1 = par1['g1_delay'] + np.random.normal(0.0, par1['g1_std'], 1)[0]
                # noise G1 contains time delay.
                # generate and retain the time additive noise in the first part of the cell cycle.
            else:
                noise_g1 = par1['g1_delay']
            # if not par1['modeltype'] in list11:  # only get noise in G2 if we don't take all noise in r.

            if par1['modeltype'] in list1 and not par1['l_std'] == 0:
                l = (1 + np.random.normal(0.0, par1['l_std'], 1)[0]) * np.log(2) / self.td

                # prevents shrinking due to a negative growth rate
                while l < 0:  # never allow a negative growth rate as this implies cells are shrinking
                    l = (1 + np.random.normal(0.0, par1['l_std'], 1)[0]) * np.log(2) / self.td

                # allows for having noise in growth rate.
            else:
                l = np.log(2)/self.td

            # G1 part of cell cycle. At this stage cell should have wb, vi, noise_g1, l and noise_thr.

            if par1['modeltype'] in list6:
                self.vi = self.wb * np.exp(noise_g1 * self.td * l) / (1 + noise_thr)
            elif par1['modeltype'] in list8:  # introduce a minimum in the volume at which cells go through Start
                self.vi = max((self.wb + noise_thr) * np.exp(noise_g1 * self.td * l), self.vb)
            elif par1['modeltype'] in list10:  # mother cells from list10 have a simple timer model
                if self.isdaughter:
                    if par1['modeltype'] in list8:
                        self.vi = max((self.wb + noise_thr) * np.exp(noise_g1 * self.td * l), self.vb)
                    else:
                        self.vi = (self.wb + noise_thr) * np.exp(noise_g1 * self.td * l)
                else:
                    if par1['modeltype'] in list8:
                        self.vi = self.vb * max(np.exp(noise_g1 * self.td * l), 1.0)
                    else:
                        self.vi = self.vb * np.exp(noise_g1 * self.td * l)
            else:
                self.vi = (self.wb + noise_thr) * np.exp(noise_g1 * self.td * l)

            # Budded part of cell cycle. At this stage should have defined growth rate l, vi, wb.
            # Volume added.
            self.vd = self.vi-1  # ensures that the following code is run at least once.

            # prevents shrinking in cells in list8. If we do this for all models then you're trying to make a positive * neg
            # = a positive and it takes forever.
            if par1['modeltype'] in list8:
                while self.vd <= self.vi:  # resample until we get a volume at division which is greater than that at birth.
                    if par1['modeltype'] in list11:
                        if par1['r_std'] == 0:
                            self.r = par1['r']
                        else:
                            self.r = par1['r']*(1+np.random.normal(0.0, par1['r_std'], 1)[0])
                        self.vd = self.vi * (1 + self.r)
                    else:
                        if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                            noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
                        else:
                            noise_g2 = 0.0
                        self.vd = self.vi * np.exp((self.CD + noise_g2) * self.td * l)
            else:
                if par1['modeltype'] in list11:
                    if par1['r_std'] == 0:
                        self.r = par1['r']
                    else:
                        self.r = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 1)[0])
                    self.vd = self.vi * (1 + self.r)
                else:
                    if par1['g2_std'] != 0:  # calculate the size and abundance of whi5 at division.
                        noise_g2 = np.random.normal(0.0, par1['g2_std'], 1)[0]
                    else:
                        noise_g2 = 0.0
                    self.vd = self.vi * np.exp((self.CD + noise_g2) * self.td * l)
            self.t_bud = np.log(self.vd / self.vi) / l  # length of time spent in budded portion of cell cycle.
            # inhibitor added. Model type determines self.delta, which is added to self.wb.
            # Noisy synthesis rate model
            if par1['modeltype'] in list2b+list12 and not par1['k_std'] == 0:  # models with noise in Whi5 synthesis rate.
                self.K = par1['K'] * (1 + np.random.normal(0.0, par1['k_std'], 1)[0]) / par1['td']  # per unit time
                # raise ValueError("I should not be here")
                # prevents shrinking
                # while self.K < 0.0:
                #     self.K = par1['K'] * (1 + np.random.normal(0.0, par1['k_std'], 1)[0]) / par1['td']  # per unit time

            if par1['modeltype'] in list2+list12:
                self.delta = self.K * self.t_bud  # noise in production of Whi5
            # Noisy integrator model. Noiseless integrator model needs no change from above.
            elif par1['modeltype'] in list3a:
                if par1['d_std'] == 0:
                    noise_d = 0.0
                else:
                    noise_d = np.random.normal(0.0, par1['d_std'], 1)[0]

                    # prevents shrinking
                    while noise_d < -1.0:
                        noise_d = np.random.normal(0.0, par1['d_std'], 1)[0]
                self.delta = self.delta * (1 + noise_d)
                del noise_d
            elif par1['modeltype'] in list2a:  # models with uncorrelated noise in Whi5 production
                if par1['d_std'] == 0:
                    noise_d = 0.0
                else:
                    noise_d = np.random.normal(0.0, par1['d_std'], 1)[0]

                    # prevents shrinking
                    while noise_d < self.delta:
                        noise_d = np.random.normal(0.0, par1['d_std'], 1)[0]
                self.delta = self.delta + noise_d
            # Set whi5 at division.
            self.wd = self.wb + self.delta

        # time of growth and input the time for the cell to divide.
        self.t_grow = np.log(self.vd / self.vb) / l
        self.t_div = self.tb + max(self.t_grow, 0)  # cells will as a minimum divide in the same timestep in
        # which they were born. Time here is measured in same units as doubling time.
        # As a consequence, self.t_div-self.tb >= self.t_grow
        if not self.mother is None:
            if self.isdaughter:  # if this cell has a mother we update the mother values to account for the new gen
                # print self.mother.gen_d
                # print self.mother.gen_m
                self.gen_d = self.mother.gen_d + 1
            elif not self.isdaughter:
                # print self.mother.gen_d
                # print self.mother.gen_m
                self.gen_m = self.mother.gen_m + 1
        Cell.cellCount += 1


def starting_popn(par1):
    # To clarify we first set the initial condition for the simulation.
    l = np.log(2)/par1['td']
    if par1['modeltype'] in [17, 18, 21, 22]:
        s_l = par1['l_std'] * l
        cd = par1['CD'] * par1['td']
        s_cd = par1['g2_std'] * par1['td']
        d = par1['delta']
        wd = 2 * par1['delta'] * (1 - exp_xy(-l, s_l, cd, s_cd))
        wm = 2 * par1['delta'] * exp_xy(- l, s_l, cd, s_cd)
        vd = d * 2 ** par1['g1_delay'] * (exp_xy(l, s_l, cd, s_cd) - 1)
        vm = d * 2 ** par1['g1_delay']
    elif par1['modeltype'] in [5, 10]:
        s_l = par1['l_std'] * l
        cd = par1['CD'] * par1['td']
        s_cd = par1['g2_std'] * par1['td']
        k = par1['K'] / par1['td']
        d = k * cd
        wd = d * (2 - exp_xy(-l, s_l, cd, s_cd)) - k * y_exp_xy(-l, s_l, cd, s_cd)
        wm = d * exp_xy(-l, s_l, cd, s_cd) + k * y_exp_xy(-l, s_l, cd, s_cd)
        vd = d * 2 ** par1['g1_delay'] * (exp_xy(l, s_l, cd, s_cd) - 1)
        vm = d * 2 ** par1['g1_delay']
    elif par1['modeltype'] in [15, 16]:
        s_l = par1['l_std'] * l
        cd = par1['CD'] * par1['td']
        s_cd = par1['g2_std'] * par1['td']
        k = par1['K'] / par1['td']
        d = k * cd
        wd = 2*d*par1['w_frac']
        wm = 2*d*(1-par1['w_frac'])
        vd = d * 2**par1['g1_delay'] * (exp_xy(l, s_l, cd, s_cd)-1)
        vm = d * 2**par1['g1_delay']
    if par1['modeltype'] in [19, 20]:
        s_l = par1['l_std'] * l
        cd = par1['CD'] * par1['td']
        s_cd = par1['g2_std'] * par1['td']
        wd = 2 * par1['delta'] * (1 - exp_xy(-l, s_l, cd, s_cd))
        wm = 2 * par1['delta'] * exp_xy(- l, s_l, cd, s_cd)
        vd = vb_func(par1, par1['g1_thresh_std'], par1['td']*par1['g2_std'])
        vm = vb_m(par1, par1['g1_thresh_std'], par1['td']*par1['g2_std'])
    if par1['modeltype'] in [23, 24]:
        d = par1['delta']
        if par1['r_std'] == 0:
            temp = par1['r']
        else:
            temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10**4))
        d_f = np.mean(temp / (1 + temp))  # unable to calculate this analytically.
        m_f = np.mean(1 / (1 + temp))
        del temp
        wd = 2 * d * d_f
        wm = 2 * d * m_f
        vd = d * par1['r']
        vm = d
        del d
    if par1['modeltype'] in [25,26]:
        d = par1['K'] * (np.log(1.0 + par1['r']) - 0.5 * par1['r_std'] ** 2 / (1 + par1['r']) ** 2) / np.log(2.0) # approx.
        if par1['r_std'] == 0:
            temp = par1['r']
        else:
            temp = par1['r'] * (1 + np.random.normal(0.0, par1['r_std'], 10**4))
        d_f = np.mean(temp / (1 + temp))  # unable to calculate this analytically.
        m_f = np.mean(1 / (1 + temp))

        wd = d * d_f+par1['K']*np.mean(np.log(1.0+temp)*temp/(1+temp))/np.log(2.0)
        wm = d * m_f+par1['K']*np.mean(np.log(1.0+temp)/(1+temp))/np.log(2.0)
        vd = d * par1['r']
        vm = d
        del d
    if par1['modeltype'] in list13:
        vd = 2*par1['r']*par1['delta']/(1+par1['r'])
        vm = 2*par1['delta']/(1+par1['r'])
        # placeholders
        wd = 0
        wm = 0

        # here the boolean optional parameter 'dilution' specifies whether the model involved
    # should be based on a Whi5 dilution scheme.
    # par=dict([('dt', 1),('nstep',100), ('td', 90), ('num_s', 100),('Vo',1.0),('std_iv',0.1),('std_iw',0.1)])
    # Initialise simulation with a normal distribution of cell sizes and Whi5 abundances.
    v_init_d = np.random.normal(loc=vd, scale=par['std_v']*vd, size=par['num_s'])
    v_init_m = np.random.normal(loc=vm, scale=par['std_v']*vm, size=par['num_s'])
    w_init_d = np.random.normal(loc=wd, scale=par['std_w']*wd, size=par['num_s'])
    w_init_m = np.random.normal(loc=wm, scale=par['std_w']*wm, size=par['num_s'])
    # print len([w_init])
    t_div = np.random.uniform(low=0.0, high=1.0, size=par['num_s'])
    # gives the fraction of the cell's doubling time
    # which this cell has passed through.
    # Now we start a list which will keep track of all currently existing cells. We will have the same list for mothers
    # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
    c = []

    for i in range(par['num_s']):    # instantiating our initial population of cells. These cells do not have mothers.
        # Daughters
        c.append(Cell(v_init_d[i], w_init_d[i], 0))
        c[-1].isdaughter = True
        c[-1].grow(par1)
        c[-1].t_div = t_div[i] * c[-1].t_div  # we expect that these cells have been caught at
        # some uniformly distributed point of progression through their cell cycles.
        c[-1].tb = c[-1].t_div-c[-1].t_grow

        # Mothers
        c.append(Cell(v_init_m[i], w_init_m[i], 0))
        c[-1].isdaughter = False
        c[-1].grow(par1)
        c[-1].t_div = t_div[i] * c[-1].t_div  # we expect that these cells have been caught at
        # some uniformly distributed point of progression through their cell cycles.
        c[-1].tb = c[-1].t_div - c[-1].t_grow
        # defined in this manner all starting cells have been born at time less than or equal to 0.
    del v_init_d, v_init_m, w_init_d, w_init_m
    return c


def starting_popn_seeded(c, par1, discr_time=False):
    # c is the set of "existent" cells at the end of the previous simulation, done so as to yield a converged
    # distribution of cells

    indices = np.random.randint(low=0, high=len(c), size=par1['num_s1'])
    temp = [c[ind] for ind in indices]
    val = []

    for obj in temp:  # copy cells this way to avoid changing the properties of the population seeded from.
        val.append(Cell(obj.vb, obj.wb, 0.0, mother=obj.mother))  # note that depending on how this code is implemented,
        # obj.mother may not exist. We therefore do the following.
        # reset all the variables to be the same as that in the previous cell, so we don't have to 'grow' them again.
        val[-1].vi = obj.vi
        val[-1].wd = obj.wd
        val[-1].vd = obj.vd
        val[-1].isdaughter = obj.isdaughter
        val[-1].t_grow = obj.t_grow
        val[-1].gen_d = obj.gen_d
        val[-1].gen_m = obj.gen_m

    if discr_time:
        nstep = par1['nstep']
        t_max = nstep * par1['dt'] * par1['td']  # maximum time of the previous simulation, so that we know what time to
        # start our new cells at. Only use this if the previous simulation was a discretized time one.
        # Otherwise start at random.
        for i0 in range(len(val)):
            val[i0].t_div = temp[i0].t_div - t_max
    else:
        t_div = np.random.uniform(low=0.0, high=1.0, size=par1['num_s1'])
        # gives the fraction of the cell's doubling time which this cell has passed through when we start tracking it.
        # Now we start a list which will keep track of all currently existing cells.
        # We will have the same list for mothers
        # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
        for i0 in range(par1['num_s1']):
            val[i0].t_div = max(t_div[i0] * val[i0].t_grow, par1['dt']*par1['td']*1.01)  # to ensure that this cell gets
            # detected in the iteration through calculating next generation cells.
            # we expect that these cells have been caught at
            # some uniformly distributed point of progression through their cell cycles.
            val[i0].tb = val[i0].t_div-val[i0].t_grow
    for i0 in range(len(val)):
        if val[i0].t_div < 0:
            raise ValueError('Error in starting_popn_seeded')
    return val


def popn_sample_storage(val):
    # val is the set of cells output by starting_popn_seeded
    output = np.zeros([3, len(val)])  # store daughter vs. mother identity, volume at birth, whi5 abundance at birth
    for i0 in range(len(val)):
        output[0, i0] = val[i0].isdaughter
        output[1, i0] = val[i0].vb
        output[2, i0] = val[i0].wb
    return output


def starting_popn_seeded_1(c, par1, discr_time=False):
    # c is the set of "existent" cells at the end of the previous simulation, done so as to yield a converged
    # distribution of cells

    indices = np.random.randint(low=0, high=len(c), size=par1['num_s1'])
    temp = [c[ind] for ind in indices]
    val = []

    for obj in temp:  # copy cells this way to avoid changing the properties of the population seeded from.
        val.append(Cell(obj.vb, obj.wb, 0.0, mother=obj.mother))  # note that depending on how this code is implemented,
        # obj.mother may not exist. We therefore do the following.
        # reset all the variables to be the same as that in the previous cell, so we don't have to 'grow' them again.
        val[-1].vi = obj.vi
        val[-1].wd = obj.wd
        val[-1].vd = obj.vd
        val[-1].isdaughter = obj.isdaughter
        val[-1].t_grow = obj.t_grow
        val[-1].gen_d = obj.gen_d
        val[-1].gen_m = obj.gen_m

    if discr_time:
        nstep = par1['nstep']
        t_max = nstep * par1['dt'] * par1['td']  # maximum time of the previous simulation, so that we know what time to
        # start our new cells at. Only use this if the previous simulation was a discretized time one.
        # Otherwise start at random.
        for i0 in range(len(val)):
            val[i0].t_div = temp[i0].t_div - t_max
            val[i0].tb = val[i0].t_div - val[i0].t_grow
    else:
        t_div = np.random.uniform(low=0.0, high=1.0, size=par1['num_s1'])
        # gives the fraction of the cell's doubling time which this cell has passed through when we start tracking it.
        # Now we start a list which will keep track of all currently existing cells.
        # We will have the same list for mothers
        # and daughters. Whether a cell is a mother or a daughter will be defined by its number of descendants.
        for i0 in range(par1['num_s1']):
            val[i0].t_div = max(t_div[i0] * val[i0].t_grow, par1['dt']*par1['td']*1.01)  # to ensure that this cell gets
            # detected in the iteration through calculating next generation cells.
            # we expect that these cells have been caught at
            # some uniformly distributed point of progression through their cell cycles.
            val[i0].tb = val[i0].t_div-val[i0].t_grow

    output = np.zeros([11, len(val)])  # return also all variables to reproduce this starting population.
    for i0 in range(len(val)):
        output[0, i0] = val[i0].vb
        output[1, i0] = val[i0].wb
        output[2, i0] = val[i0].tb
        output[3, i0] = val[i0].vi
        output[4, i0] = val[i0].wd
        output[5, i0] = val[i0].vd
        output[6, i0] = val[i0].isdaughter
        output[7, i0] = val[i0].t_grow
        output[8, i0] = val[i0].gen_d
        output[9, i0] = val[i0].gen_m
        output[10, i0] = val[i0].t_div

    for i0 in range(len(val)):
        if val[i0].t_div < 0:
            raise ValueError('Error in starting_popn_seeded')
    return val, output


def from_stored_pop(pop, par1):
    # This function behaves differently depending on whether your stored population either is from pop_sample_storage
    # or from starting_popn_seeded_1. It decides this based on the length of the vector.
    val = []  # list of new cells being formed.
    if pop.shape[0] == 3:
        # in this case we assume that it came from pop_sample_storage. We just grow these cells again.
        # output[0, i0] = val[i0].isdaughter
        # output[1, i0] = val[i0].vb
        # output[2, i0] = val[i0].wb
        t_div = np.random.uniform(low=0.0, high=1.0, size=pop.shape[1])  # seeds these cells at a random time through
        # their cell cycles.
        for i0 in range(pop.shape[1]):
            val.append(Cell(pop[1, i0], pop[2, i0], 0.0, mother=None))  # here mother is just set to be None.
            val[-1].isdaughter = pop[0, i0]
            val[-1].grow(par1)
            if __name__ == '__main__':
                val[-1].t_div = t_div[i0]*val[-1].t_div  # rescale the division time for this cell
                val[-1].tb = val[-1].t_div - val[-1].t_grow  # adjusts their birth time accordingly

    else:
        # in this case we assume that it came from starting_popn_seeded_1
        # output[0, i0] = val[i0].vb
        # output[1, i0] = val[i0].wb
        # output[2, i0] = val[i0].tb
        # output[3, i0] = val[i0].vi
        # output[4, i0] = val[i0].wd
        # output[5, i0] = val[i0].vd
        # output[6, i0] = val[i0].isdaughter
        # output[7, i0] = val[i0].t_grow
        # output[8, i0] = val[i0].gen_d
        # output[9, i0] = val[i0].gen_m
        # output[10, i0] = val[i0].t_div
        for i0 in range(pop.shape[1]):  #
            #  copy cells this way to avoid changing the properties of the population seeded from.
            val.append(Cell(pop[0, i0], pop[1, i0], pop[2, i0], mother=None)) # here mother is just set to be None.
            # reset all the variables to be the same as that in the previous cell, so we don't have to 'grow' them
            # again.
            val[-1].vi = pop[3, i0]
            val[-1].wd = pop[4, i0]
            val[-1].vd = pop[5, i0]
            val[-1].isdaughter = pop[6, i0]
            val[-1].t_grow = pop[7, i0]
            val[-1].gen_d = pop[8, i0]
            val[-1].gen_m = pop[9, i0]
            val[-1].t_div = pop[10, i0]
    return val


def next_gen(index, f, t, par1):
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    # frac = max((f[index].vd-f[index].vi)/f[index].vd, 0.0)
    if par1['modeltype'] in list13:
        # frac1 = par1['r']/(1+par1['r'])  # non-budding
        frac1 = (f[index].vd - f[index].vi) / f[index].vd  # budding
    if par1['modeltype'] in list4:
        frac1 = 1-np.exp(-par1['CD']*np.log(2))
        # model 2 has noise distributed volumetrically between m and d.
    if par1['modeltype'] in list5:
        frac1 = (f[index].vd-f[index].vi)/f[index].vd
        # for now we will assume that Whi5 is distributed exactly volumetrically.
    if par1['modeltype'] in list7:
        frac1 = par1['frac']  # Constant division ratio 'frac'
        # raise ValueError('you be good now')
    if par1['modeltype'] in list9:  # daughter cells receive constant fraction of Whi5
        frac2 = par1['w_frac']
    else:
        frac2 = frac1
    f.append(Cell(frac1*f[index].vd, frac2*f[index].wd, t, mother=weakref.proxy(f[index])))
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].isdaughter = True
    f[-1].grow(par1)  # grow newborn cell
    f[index].daughter = weakref.proxy(f[-1])  # Update the mother cell to show this cell as a daughter.
    # add new cell for newborn mother cell.
    f.append(Cell((1-frac1)*f[index].vd, (1-frac2)*f[index].wd, t, mother=weakref.proxy(f[index])))
    f[-1].grow(par1)  # grow newborn cell
    f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
    f[index].exists = False  # track that this cell no longer "exists".
    return f


def discr_time(par1):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a set of parameters par1.
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'] * par1['td'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    num_div_d = np.zeros(tvec.shape)  # keep track of the number of divisions from mother cells
    num_div_m = np.zeros(tvec.shape)  # keep track of the number of divisions from daughter cells
    # Define lists which will keep track of the time step in which each cell divides.
    div_times = []
    for i in range(nstep+1):
        div_times.append([])
    # Now we go through our starting population and determine at which time step they will divide (cells with a division
    # of all cells and store that.
    c = starting_popn(par1)
    num_cells[0] = len(c)
    for i in range(len(c)):
        if c[i].t_div < np.amax(tvec):  # ensures we never consider cells which would go into the (nstep+2)th bin, since
            # there is none in this setup
            td_ind = np.searchsorted(tvec, np.array(c[i].t_div), side='left', sorter=None)
            # left means that e.g. (n-1)*dt<x<=n*dt goes to the nth bin.
            # nstep*dt<x means that the index is (nstep+1).
            div_times[td_ind].append(i)
            # a cell is picked to divide in the timestep after it divides, i.e. 11 if divides in the 11th time interval
            del td_ind
    # Now we begin iterating through the time values in tvec
    for i in range(nstep+1):
        if i > 0:
            num_div_d[i] = num_div_d[i-1]
            num_div_m[i] = num_div_m[i-1]
        for index in div_times[i]:  # the presence of indices in the 0th timestep means that the seeding cells grew
            # negatively. Statistically unlikely but may happen for one or two cells.

            # Double check that these indices have been allocated correctly: c[index].t_div should be in the interval
            # (tvec[i-1], tvec[i]]
            if i != 0:
                if tvec[i-1] >= c[index].t_div or tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i+1]
                    raise ValueError('The cell division orders are wrong')
            elif i == 0:
                if tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i+1]
                    raise ValueError('The cell division orders are wrong')

            c = next_gen(index, c, c[index].t_div, par1)  # compute the next generation for this cell.
            if c[index].isdaughter:
                num_div_d[i] += 1
            elif not c[index].isdaughter:
                num_div_m[i] += 1
            for j in range(2):  # Set the time of division for the two new cells produced by next_gen
                t_div = c[len(c) - 1 - j].t_div
                if t_div < np.amax(tvec):  # We only mark cells for division if they fall within the time frame of our
                    # simulation
                    c[len(c) - 1 - j].should_div = True  # this cell should for sure divide at some stage within the sim
                    # simulation
                    td_ind = np.searchsorted(tvec, np.array(t_div), side='left', sorter=None)
                    if td_ind < i:
                        print "Timestep", i, "div timestep", td_ind, "t_div", t_div, "t_birth", c[len(c)-1-j].tb, "t_div_prev", c[index].t_div
                        raise ValueError('Cells are falling behind')
                    div_times[td_ind].append(len(c) - 1 - j)
                    # a cell is picked to divide in the time step before it divides, i.e. 10 if divides in the 11th time
                    # interval
                    del td_ind
                del t_div
        num_cells[i] = len(c)
    obs = [num_cells, tvec, num_div_d, num_div_m]
    return c, obs


def discr_time_1(par1, starting_pop):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a set of parameters par1 and a starting population of cells.
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'] * par1['td'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    num_div_d = np.zeros(tvec.shape)  # keep track of the number of divisions from mother cells
    num_div_m = np.zeros(tvec.shape)  # keep track of the number of divisions from daughter cells
    # Define lists which will keep track of the time step in which each cell divides.
    div_times = []
    for i in range(nstep + 1):
        div_times.append([])
    # Now we go through our starting population and determine at which time step they will divide (cells with a division
    # of all cells and store that.
    c = starting_pop[:]
    num_cells[0] = len(c)
    for i in range(len(c)):
        if c[i].t_div < np.amax(tvec):  # ensures we never consider cells which would go into the (nstep+2)th bin, since
            # there is none in this setup
            td_ind = np.searchsorted(tvec, np.array(c[i].t_div), side='left', sorter=None)
            # left means that e.g. (n-1)*dt<x<=n*dt goes to the nth bin.
            # nstep*dt<x means that the index is (nstep+1).
            div_times[td_ind].append(i)
            # a cell is picked to divide in the timestep after it divides, i.e. 11 if divides in the 11th time interval
            del td_ind
    # Now we begin iterating through the time values in tvec
    for i in range(nstep + 1):
        if i > 0:
            num_div_d[i] = num_div_d[i - 1]
            num_div_m[i] = num_div_m[i - 1]
        for index in div_times[i]:  # the presence of indices in the 0th timestep means that the seeding cells grew
            # negatively. Statistically unlikely but may happen for one or two cells.

            # Double check that these indices have been allocated correctly: c[index].t_div should be in the interval
            # (tvec[i-1], tvec[i]]
            if i != 0:
                if tvec[i - 1] >= c[index].t_div or tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i + 1]
                    raise ValueError('The cell division orders are wrong')
            elif i == 0:
                if tvec[i] < c[index].t_div:
                    print "timestep", i, "Cell index", index, "Cell T_div", c[index].tdiv, "Bin start", \
                        tvec[i], "Bin end", tvec[i + 1]
                    raise ValueError('The cell division orders are wrong')

            c = next_gen(index, c, c[index].t_div, par1)  # compute the next generation for this cell.
            if c[index].isdaughter:
                num_div_d[i] += 1
            elif not c[index].isdaughter:
                num_div_m[i] += 1
            for j in range(2):  # Set the time of division for the two new cells produced by next_gen
                t_div = c[len(c) - 1 - j].t_div
                if t_div < np.amax(tvec):  # We only mark cells for division if they fall within the time frame of our
                    # simulation
                    c[len(c) - 1 - j].should_div = True  # this cell should for sure divide at some stage within the sim
                    # simulation
                    td_ind = np.searchsorted(tvec, np.array(t_div), side='left', sorter=None)
                    if td_ind < i:
                        raise ValueError('Cells are falling behind')
                    div_times[td_ind].append(len(c) - 1 - j)
                    # a cell is picked to divide in the time step before it divides, i.e. 10 if divides in the 11th time
                    # interval
                    del td_ind
                del t_div
        num_cells[i] = len(c)
    obs = [num_cells, tvec, num_div_d, num_div_m]
    return c, obs


def discr_gen(par1):
    #  This discretized generation simulation will be used to test whether observed deviations from expected values
    #  in population growth simulations (with discretized time) arise as a result of differences in the distributions
    #  being sampled from in a population with "competitive" growth, or simply from the math being wrong. Note that
    #  no attention should be payed to the relative timing order of things here.
    num_gen = par1['num_gen']
    c = starting_popn(par1)  # gives us the starting population in the same way as with discretized time
    for i in range(num_gen):
        temp = len(c)
        for index in range(temp):
            if c[index].exists:
                c = next_gen(index, c, i+1, par1)
                # iterate through c to produce a new mother daughter pair for each cell
        del temp
    return c


def discr_gen_1(par1, starting_pop):
    #  This discretized generation simulation will be used to test whether observed deviations from expected values
    #  in population growth simulations (with discretized time) arise as a result of differences in the distributions
    #  being sampled from in a population with "competitive" growth, or simply from the math being wrong. Note that
    #  no attention should be payed to the relative timing order of things here.
    num_gen = par1['num_gen1']  # note that this allows this function to be called independently of discr_gen.
    c = starting_pop[:]  # gives us the starting population in the same way as with discretized time
    for i in range(num_gen):
        r = len(c)
        for index in range(r):
            if c[index].exists:
                c = next_gen(index, c, i+1, par1)
                # iterate through c to produce a new mother daughter pair for each cell
    return c

########################################################################################################################
# SINGLE PARAMETER SET MEASUREMENT CODE
########################################################################################################################


def single_par_meas(par1):
    c, temp = discr_time(par1)
    obs = np.empty([8, 2])
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.exists]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs[0,k] = vals[0]
        del vals
        obs[1,k] = np.mean(vb)
        obs[2,k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3,k] = np.mean(vd)
        obs[4,k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5,k] = np.mean(wb)
        obs[6,k] = num_neg
        obs[7,k] = num_cell
    return obs


def single_par_meas2(par1):
    c = discr_gen(par1)
    obs = np.empty([8, 2])
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        num_lc = len([obj for obj in c[1000:] if obj.isdaughter == k and obj.wd / obj.vd < 1.0])*100.0/len(vb)
        vals = scipy.stats.linregress(vb, vd)
        obs[0, k] = vals[0]
        del vals
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
        obs[6, k] = np.mean(np.asarray(wb)*np.asarray(wb))
        obs[7, k] = num_lc
    del c
    return obs


def single_par_meas3(par1):
    c, num_cells, tvec, av_whi5, num_div_d, num_div_m = discr_time(par1)
    obs = np.empty([8, 2])
    obs1 = np.empty([8, 2])
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.exists]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs[0,k] = vals[0]
        del vals
        obs[1,k] = np.mean(vb)
        obs[2,k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3,k] = np.mean(vd)
        obs[4,k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5,k] = np.mean(wb)
        obs[6,k] = num_neg
        obs[7,k] = num_cell

        vb = [obj.vb for obj in c[1000:] if
              obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs1[0, k] = vals[0]
        del vals
        obs1[1, k] = np.mean(vb)
        obs1[2, k] = np.mean(np.asarray(vb) * np.asarray(vb))
        obs1[3, k] = np.mean(vd)
        obs1[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs1[5, k] = np.mean(wb)
        obs1[6, k] = num_neg
        obs1[7, k] = num_cell
    return obs, obs1


def single_par_meas4(par1):  # same as single_par_meas2 except that it also returns the distribution of growth times
    c = discr_gen(par1)
    obs = np.empty([6, 2])
    t_grow = []
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        t_grow.append(np.asarray([obj.t_grow for obj in c[1000:] if obj.isdaughter == k]))
        # mothers first, then daughters
        vals = scipy.stats.linregress(vb, vd)
        obs[0, k] = vals[0]
        del vals
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
    del c
    return obs, t_grow


def single_par_meas5(par1):  # same as single_par_meas3 except that it also returns the distribution of growth times
    c, num_cells, tvec, av_whi5, num_div_d, num_div_m = discr_time(par1)
    obs = np.empty([8, 2])
    obs1 = np.empty([8, 2])
    t_grow = []
    t_grow1 = []
    if len(c) < 1000:
        raise ValueError('error in parameter set CD='+str(np.round(par1['CD'], 2))+' g1_std='
                         + str(np.round(par1['g1_thresh_std'], 2))+' g2_std'+str(np.round(par1['g2_std'], 2)))
    for k in range(2):
        vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k and obj.exists]
        t_grow.append(np.asarray([obj.t_grow for obj in c[1000:] if obj.isdaughter == k and obj.exists]))

        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs[0, k] = vals[0]
        del vals
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
        obs[6, k] = num_neg
        obs[7, k] = num_cell

        vb = [obj.vb for obj in c[1000:] if
              obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
        vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
        wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        t_grow1.append(np.asarray([obj.t_grow for obj in c[1000:] if obj.isdaughter == k]))
        num_neg = np.sum([obj.vd < obj.vb for obj in c[1000:] if obj.isdaughter == k])  # number of cells with neg tgrow
        num_cell = len([obj for obj in c[1000:] if obj.isdaughter == k])  # number of cells of each cell type
        vals = scipy.stats.linregress(vb, vd)
        obs1[0, k] = vals[0]
        del vals
        obs1[1, k] = np.mean(vb)
        obs1[2, k] = np.mean(np.asarray(vb) * np.asarray(vb))
        obs1[3, k] = np.mean(vd)
        obs1[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs1[5, k] = np.mean(wb)
        obs1[6, k] = num_neg
        obs1[7, k] = num_cell
    return obs, obs1, t_grow, t_grow1


def single_par_meas6(par1):  # same as single_par_meas2 except that it also returns the distribution of growth times
    c = discr_gen(par1)
    obs = np.empty([6, 3])
    for k in range(3):
        if k < 2:
            vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
            vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
            wb = [obj.wb for obj in c[1000:] if obj.isdaughter == k]
        elif k == 2:
            vb = [obj.vb for obj in c[1000:]]  # mothers are at 0, then daughters are at 1.
            vd = [obj.vd for obj in c[1000:]]
            wb = [obj.wb for obj in c[1000:]]
        # mothers first, then daughters
        val = scipy.stats.linregress(vb, vd)
        obs[0, k] = val[0]
        del val
        obs[1, k] = np.mean(vb)
        obs[2, k] = np.mean(np.asarray(vb)*np.asarray(vb))
        obs[3, k] = np.mean(vd)
        obs[4, k] = np.mean(np.asarray(vd) * np.asarray(vb))
        obs[5, k] = np.mean(wb)
    del c
    return obs  # returns mothers, daughters, and whole population respectively.


def heat_maps(obs, labels, g1_std, g2_std):
    font = {'family': 'normal', 'weight': 'bold', 'size': 12}
    plt.rc('font', **font)
    model = ['dilution symmetric', 'initiator symmetric']
    plots = [0, 1, 3, 4]
    figs=[]
    for i in range(len(plots)):
        for j in range(obs[0].shape[2]):
            figs.append(plt.figure(figsize=[11, 10]))
            sns.heatmap(obs[plots[i]][:, :, j], xticklabels=np.around(g2_std, decimals=2), \
                             yticklabels=np.around(g1_std[::-1], decimals=2), annot=True)
            plt.xlabel('C+D timing noise $\sigma_{C+D}$',size=20)
            plt.ylabel('Start threshold noise $\sigma_{thresh}$', size=20)
            plt.title(labels[plots[i]]+' '+model[j], size=20)
    return figs


def heat_maps_mother_daughter(obs, g1_std, g2_std, model):
    # Assumes that mothers come first in the third dimension of obs
    label = ['Mothers', 'Daughters']
    figs = []
    for i in range(2):
        figs.append(plt.figure(figsize=[16, 15]))
        sns.heatmap(obs[:, :, 0, i], xticklabels=np.around(g2_std, decimals=2),
                    yticklabels=np.around(g1_std[::-1], decimals=2), annot=True)
        plt.xlabel('C+D timing noise $\sigma_{C+D}$', size=20)
        plt.ylabel('Start threshold noise $\sigma_{thresh}$', size=20)
        plt.title(label[i]+' '+model, size=20)
    print len(figs)
    return figs


def heat_map(obs, y, x, ax, xlabel=None, ylabel=None, title=None, fmt='.2g'):
    # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
    # will be modified.
    plt.sca(ax)
    sns.heatmap(obs[::-1, :], xticklabels=np.around(x, decimals=2),
                yticklabels=np.around(y[::-1], decimals=2), annot=False)
    if xlabel:
        ax.set_xlabel(xlabel, size=20)
    if ylabel:
        ax.set_ylabel(ylabel, size=20)
    if title:
        ax.set_title(title, size=20)
    plt.xticks(size=14)
    plt.yticks(size=14)
    ax.tick_params(labelsize=14)
    return ax


def heat_map_1(obs, y, x, ax, xlabel=None, ylabel=None, title=None, bound=0.1, val=1.0, color='black', outline=True,
               alpha=1.0, cmap_lims=None):
    # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
    # will be modified.

    # Note that if an outline is desired, then the outline must not include the entire figure, otherwise an
    # AssertionError is thrown.
    plt.sca(ax)
    if cmap_lims is None:
        sns.heatmap(obs[::-1, :], cmap="coolwarm", xticklabels=np.around(x, decimals=2),
                    yticklabels=np.around(y[::-1], decimals=2), annot=False, vmin=max(np.round(np.amin(obs), 2), 0.0),
                    vmax=np.round(np.amax(obs), 2))
    else:
        sns.heatmap(obs[::-1, :], cmap="coolwarm", xticklabels=np.around(x, decimals=2),
                    yticklabels=np.around(y[::-1], decimals=2), annot=False, vmin=cmap_lims[0],
                    vmax=cmap_lims[1])
    # print np.round(np.amin(obs), 2), np.round(np.amax(obs), 2)
    # for i0 in range(obs.shape[0])
    if outline:
        temp1 = np.transpose(obs[:, :])
        ind2 = 0
        for ind0 in range(temp1.shape[0]):
            for ind1 in range(temp1.shape[1]):
                if val - bound <= temp1[ind0, ind1] <= val + bound:
                    temp2 = LineString(
                        [(ind0, ind1), (ind0 + 1, ind1), (ind0 + 1, ind1 + 1), (ind0, ind1 + 1), (ind0, ind1)])
                    if ind2 == 1:
                        temp3 = temp3.symmetric_difference(temp2)
                    else:
                        temp3 = temp2
                        ind2 += 1
        if ind2 == 1:
            dil = temp3.buffer(0.02)
            patch = PolygonPatch(dil, facecolor=color, edgecolor=color, alpha=alpha)
            ax.add_patch(patch)
    if xlabel:
        # ax.set_xlabel(xlabel, size=20)
        ax.set_xlabel(xlabel)
    if ylabel:
        # ax.set_ylabel(ylabel, size=20)
        ax.set_ylabel(ylabel)
    if title:
        # ax.set_title(title, size=20)
        ax.set_title(title)
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.tick_params(labelsize=14)
    return ax


def heat_map_pd(obs, y, x, ax, xlabel=None, ylabel=None, title=None, bound=0.1, val=1.0, color='black', outline=True,
                   alpha=1.0, cmap_lims=None, n=5):
    # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
    # will be modified.

    # Note that if an outline is desired, then the outline must not include the entire figure, otherwise an
    # AssertionError is thrown.
    plt.sca(ax)

    y_lab = pd.Series(y[::-1])
    x_lab = pd.Series(x)
    temp = pd.DataFrame(obs[::-1, :], index=y_lab, columns=x_lab)

    if cmap_lims is None:
        sns.heatmap(temp, cmap="coolwarm", xticklabels=n,
                    yticklabels=n, annot=False, vmin=max(np.round(np.amin(obs), 2), 0.0),
                    vmax=np.round(np.amax(obs), 2))
    else:
        sns.heatmap(temp, cmap="coolwarm", xticklabels=n,yticklabels=n, annot=False, vmin=cmap_lims[0],
                    vmax=cmap_lims[1])
    # print np.round(np.amin(obs), 2), np.round(np.amax(obs), 2)
    # for i0 in range(obs.shape[0])
    if outline:
        temp1 = np.transpose(obs[:, :])
        a = np.absolute(temp1 - val) < bound  # gives a binary array
        b=scipy.ndimage.morphology.binary_fill_holes(a)
        c=scipy.ndimage.filters.median_filter(b, size=(3, 3))
        nums = measure.label(c)
        # if np.amin(nums)!=np.amax(nums):  # in this case we don't produce an outline since the whole plot is the same
        temp3=[]
        for ind in range(np.amax(nums)):
            temp3.append([])
        print len(temp3)
        print np.amax(nums)
        for ind0 in range(temp1.shape[0]):
            for ind1 in range(temp1.shape[1]):
                if c[ind0, ind1]!=0:
                    temp2 = LineString(
                        [(ind0, ind1), (ind0 + 1, ind1), (ind0 + 1, ind1 + 1), (ind0, ind1 + 1), (ind0, ind1)])
                    temp5=nums[ind0, ind1]

                    temp_var = np.zeros(4)  # check that this is part of a connected set
                    if ind0 != 0:
                        if nums[ind0-1, ind1]==temp5:
                            temp_var[0] = 1
                    if ind0 != temp1.shape[0] - 1:
                        if nums[ind0 + 1, ind1] == temp5:
                            temp_var[1] = 1
                    if ind1 != 0:
                        if nums[ind0, ind1-1] == temp5:
                            temp_var[2] = 1
                    if ind1 != temp1.shape[1] - 1:
                        if nums[ind0, ind1 + 1] == temp5:
                            temp_var[3] = 1
                    if np.sum(temp_var)>0:  # We only add this one if it is edgewise connected, not cornerwise
                        if len(temp3[temp5-1])==0:
                            temp3[temp5-1].append(temp2)
                        else:
                            temp3[temp5-1][0]=temp3[temp5-1][0].symmetric_difference(temp2)



                    # indices = [[ind0 - 1, ind1], [ind0 + 1, ind1], [ind0, ind1 - 1], [ind0, ind1 + 1]]
                    # temp5=np.zeros(4)
                    # for temp_ind in range(4):
                    #     if temp4[indices[temp_ind][0], indices[temp_ind][1]]>=0:
                    #         temp5[temp_ind]=1
                    # if np.sum(temp5)==0:
                    #     temp4[ind0, ind1] = len(temp3)
                    #     temp3.append(temp2)
                    # else:
                    #     temp6=np.nonzero(temp5)
                    #     temp4[ind0, ind1]=temp4[indices[temp6[0]][0], indices[temp6[0]][0]]
                    #     temp3[]=temp3[temp4[indices[temp6[0]][0], indices[temp6[0]][0]]].symmetric_difference(temp2)
                    #     temp3 = temp3.symmetric_difference(temp2)
                    #     temp3 = temp2
                    #     ind2 += 1
        for ind in range(len(temp3)):
            if len(temp3[ind]) != 0:
                dil = temp3[ind][0].buffer(0.02)
                patch = PolygonPatch(dil, facecolor=color, edgecolor=color, alpha=alpha)
                ax.add_patch(patch)
    if xlabel:
        # ax.set_xlabel(xlabel, size=20)
        ax.set_xlabel(xlabel)
    if ylabel:
        # ax.set_ylabel(ylabel, size=20)
        ax.set_ylabel(ylabel)
    if title:
        # ax.set_title(title, size=20)
        ax.set_title(title)
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.tick_params(labelsize=14)
    return ax


def test_initiation_times(c, label):
    t_vec = [obj.td*np.log(obj.vi/obj.vb)/np.log(2) for obj in c]
    plt.figure(figsize=[6,6])
    sns.distplot(t_vec,label='fraction negative ='+str(round(np.sum([obj < 0 for obj in t_vec])*1.0/len(t_vec),2)))
    plt.title('Histogram of initiation times'+label)
    plt.xlabel('initiation times')
    plt.legend()
    return t_vec


def test_budding_times(c):
    t_vec = [obj.td*np.log(obj.vd/obj.vi)/np.log(2) for obj in c]
    plt.figure(figsize=[6, 6])
    sns.distplot(t_vec, label='fraction negative ='+str(round(np.sum([obj < 0 for obj in t_vec])*1.0/len(t_vec),2)))
    plt.title('Histogram of budding times')
    plt.xlabel('initiation times')
    plt.legend()
    return t_vec


def test_division_times(c, asymm=False):
    if asymm:
        t_vec=[]
        plt.figure(figsize=[6, 6])
        celltype=['mother','daughter']
        for k in range(2):
            t_vec.append([obj.t_grow for obj in c if obj.isdaughter==k])
            sns.distplot(t_vec[k],label=celltype[k]+' fraction negative ='+str(round(np.sum([obj < 0 for obj in t_vec])*1.0/len(t_vec),2)))
        plt.title('Histogram of interdivision times')
        plt.xlabel('interdivision times (td=1)')
        plt.legend()
    else:
        plt.figure(figsize=[6, 6])
        t_vec=[obj.t_grow for obj in c]
        sns.distplot(t_vec,label='fraction negative ='+str(round(np.sum([obj < 0 for obj in t_vec])*1.0/len(t_vec),2)))
        plt.title('Histogram of interdivision times')
        plt.xlabel('interdivision times (td=1)')
        plt.legend()
    return t_vec


########################################################################################################################
# This is to test the functional form of the correlation between Vb and Vd for daughter cells with asymmetric growth
# and no noise in the production of Whi5. Works with obs being the output of parameter_variation_asymmetric
########################################################################################################################
def exp_xy(l, s_l, cd, s_cd):
    exy = np.exp(0.5*((l*s_cd)**2+(cd*s_l)**2+2*l*cd)/(1-s_l**2*s_cd**2))/np.sqrt(1-s_l**2*s_cd**2)
    return exy


def y_exp_xy(l, s_l, cd, s_cd):
    val = np.exp((2 * cd * l + l ** 2 * s_cd ** 2 + cd ** 2 * s_l ** 2) / (2 - 2 * s_cd ** 2 * s_l ** 2)) * (
        cd + l * s_cd ** 2) / (1 - s_cd ** 2 * s_l ** 2) ** (3 / 2)
    return val


def yy_exp_xy(l, s_l, cd, s_cd):
    val = np.exp((2 * cd * l + l ** 2 * s_cd ** 2 + cd ** 2 * s_l ** 2) / (2 - 2 * s_cd ** 2 * s_l ** 2)) * (s_cd ** 2
            + (cd + l * s_cd ** 2) ** 2 - s_cd ** 4 * s_l ** 2) / (1 - s_cd ** 2*s_l ** 2) ** (5 / 2)
    return val


def tay_e1(s1):
    return 1 + s1**2 + s1**4 + s1**6 + s1**8


def tay_e2(s1):
    return 1 + 3*s1**2 + 5*s1**4 + 7*s1**6 + 9*s1**8


def wbwb_p(par1, s_i, s_cd):  # units should be volume and time for s_i and s_cd respectively.
    td = par1['td']
    cd = par1['CD'] * par1['td']  # units of time
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k*cd
    r = np.exp(l * cd) - 1
    if par1['modeltype'] == 1:
        a = 1 - 2 * np.exp(0.5 * (l * s_cd) ** 2) / (1 + r) + 2 * np.exp(2 * (l * s_cd) ** 2) / (1 + r) ** 2
        val = (3 * d**2*a + k**2*s_cd**2+8*d*k*l*s_cd**2*(np.exp(0.5*s_cd**2*l**2)-2*np.exp(2*l**2*s_cd**2)/(1+r))/
                (1+r)+2*k**2*s_cd**2*(-np.exp(0.5*s_cd**2*l**2)*(1+l**2*s_cd**2)+np.exp(2*s_cd**2*l**2)*
                                      (1+4*l**2*s_cd**2)/(1+r))/(1+r))/(2-a)
    if par1['modeltype'] in [5, 10]:
        s_l = par1['l_std'] * l
        s_k = par1['k_std'] * k  # units of K
        a = 1 - 2 * exp_xy(-l, s_l, cd, s_cd) + 2 * exp_xy(-2 * l, 2 * s_l, cd, s_cd)
        val = (2-a) ** (-1) * (3 * d ** 2 + 4 * d * k * y_exp_xy(-2*l, 2*s_l, cd, s_cd) - 4 * d * k * \
                             y_exp_xy(-l, s_l, cd, s_cd) + s_k ** 2 * (cd ** 2 + s_cd ** 2) + k ** 2 * s_cd ** 2 - 2 \
                            * (k ** 2 + s_k ** 2) * yy_exp_xy(-l, s_l, cd, s_cd) + 2 * (k ** 2 + s_k ** 2) \
                                                                                * yy_exp_xy(-2 * l, 2 * s_l, cd, s_cd))
    if par1['modeltype'] in [15, 16]:
        s_l = par1['l_std'] * l
        s_k = par1['k_std'] * k  # units of K
        a = 1 - 2*par1['w_frac'] + 2*par1['w_frac']**2
        val = (2*d**2+(k**2+s_k**2)*(cd**2+s_cd**2))*a/(2-a)
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        s_l = par1['l_std'] * l
        a = 1-2*exp_xy(-l, s_l, cd, s_cd)+2*exp_xy(-2*l, 2*s_l, cd, s_cd)
        val = d ** 2 * a * (3 + par1['d_std']**2) / (2 - a)
    return val


def vbvb_func(par1, s_i, s_cd):
    # Model number 0
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k*cd
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        a = 1 - 2 * np.exp(0.5 * (l * s_cd) ** 2) / (1 + r) + 2 * np.exp(2 * (l * s_cd) ** 2) / (1 + r) ** 2
        val = ((1+r)**2*np.exp(2*(l*s_cd)**2)-2*(1+r)*np.exp(0.5*(l*s_cd)**2)+1)*(3*a*d**2/(2-a)+s_i**2)
    if par1['modeltype'] == 1:
        a = 1 - 2 * np.exp(0.5 * (l * s_cd) ** 2) / (1 + r) + 2 * np.exp(2 * (l * s_cd) ** 2) / (1 + r) ** 2
        temp = (3 * d**2*a + k**2*s_cd**2+8*d*k*l*s_cd**2*(np.exp(0.5*s_cd**2*l**2)-2*np.exp(2*l**2*s_cd**2)/(1+r))/
                (1+r)+2*k**2*s_cd**2*(-np.exp(0.5*s_cd**2*l**2)*(1+l**2*s_cd**2)+np.exp(2*s_cd**2*l**2)*
                                      (1+4*l**2*s_cd**2)/(1+r))/(1+r))/(2-a)
        val = (temp+s_i**2)*((1+r)**2*np.exp(2*(l*s_cd)**2)-2*(1+r)*np.exp(0.5*(l*s_cd)**2)+1)
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  # noiseless adder + noise in growth rate
        s_l = par1['l_std']*l
        a = 1-2*exp_xy(-l, s_l, cd, s_cd)+2*exp_xy(-2*l, 2*s_l, cd, s_cd)

        val = (3*d**2*a/(2-a)+s_i**2)*(exp_xy(2*l, 2*s_l, cd, s_cd)-2*exp_xy(l, s_l, cd, s_cd)+1)
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = wbwb_p(par1, s_i, s_cd)
        val = (exp_xy(2*l, 2*s_l, cd, s_cd)-2*exp_xy(l, s_l, cd, s_cd)+1)*(val+s_i**2)
    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        s_l = par1['l_std'] * l
        a = 1 - 2 * exp_xy(-l, s_l, cd, s_cd) + 2 * exp_xy(-2 * l, 2 * s_l, cd, s_cd)
        val = (3 * d ** 2 * a / (2 - a)) * (
            exp_xy(2 * l, 2 * s_l, cd, s_cd) - 2 * exp_xy(l, s_l, cd, s_cd) + 1) * tay_e2(s_i)
    if par1['modeltype'] in [15, 16]:
        s_l = par1['l_std'] * l
        val = (exp_xy(2*l, 2*s_l, cd, s_cd)-2*exp_xy(l, s_l, cd, s_cd)+1)*(wbwb_p(par1, s_i, s_cd) + s_i**2)
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        s_l = par1['l_std'] * l
        val = (wbwb_p(par1, s_i, s_cd)+(d*s_i)**2)*(exp_xy(2*l, 2*s_l, cd, s_cd)-2*exp_xy(l, s_l, cd, s_cd)+1)
    if par1['modeltype'] in [19, 20]:  # different model for mothers vs. daughters
        del d
        d = par1['delta']
        s_l = par1['l_std'] * l
        s_new = np.sqrt(par1['g1_std'] ** 2 + par1['g2_std'] ** 2) * par1['td']
        s_new1 = np.sqrt(4*par1['g1_std'] ** 2 + par1['g2_std'] ** 2) * par1['td']
        val = (exp_xy(2*l, 2*s_l, par1['g1_delay']*par1['td'], par1['g1_std']*par1['td']) -
               2*exp_xy(l, s_l, (2*par1['g1_delay']+par1['CD'])*par1['td'], s_new1) +
            exp_xy(2*l, 2*s_l, (par1['g1_delay']+par1['CD'])*par1['td'], s_new))*(wbwb(par1, cell_no=1)+(d*s_i)**2)/\
              (2-exp_xy(2*l, 2*s_l, par1['g1_delay']*par1['td'], par1['g1_std']*par1['td']))
    return val


def vb_func(par1, s_i, s_cd):  # daughter cells
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k * cd
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        val = d * ((1 + r) * np.exp(0.5 * (l * s_cd) ** 2) - 1)
    if par1['modeltype'] == 1:
        val = d * ((1 + r) * np.exp(0.5 * (l * s_cd) ** 2) - 1)
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  # noiseless adder + noise in growth rate
        s_l = par1['l_std'] * l
        val = d * (exp_xy(l, s_l, cd, s_cd) - 1)
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        # noisy Whi5 adder + noise in growth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = d * (exp_xy(l, s_l, cd, s_cd) - 1)
    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        s_l = par1['l_std'] * l
        val = d * (exp_xy(l, s_l, cd, s_cd) - 1)*tay_e1(s_i)
    if par1['modeltype'] in [15, 16]:
        s_l = par1['l_std'] * l
        val = d * (exp_xy(l, s_l, cd, s_cd) - 1)
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        s_l = par1['l_std'] * l
        val = d * (exp_xy(l, s_l, cd, s_cd) - 1)
    if par1['modeltype'] in [19, 20]:  # different model for mothers vs. daughters
        s_l = par1['l_std'] * l
        s_new = np.sqrt(par1['g1_std']**2+par1['g2_std']**2)*par1['td']

        val = (exp_xy(l, s_l, (par1['g1_delay']+par1['CD'])*par1['td'], s_new) -
               exp_xy(l, s_l, par1['g1_delay']*par1['td'], par1['g1_std'] * par1['td'])) \
             * wb(par1, cell_no=1) / (2 - exp_xy(l, s_l, par1['g1_delay']*par1['td'], par1['g1_std']*par1['td']))
    return val


def vd_func(par1, s_i, s_cd):
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k * cd
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        val = 2 * d * (1 + r - np.exp(0.5 * (l * s_cd) ** 2)) * np.exp(0.5 * (l * s_cd) ** 2)
    if par1['modeltype'] == 1:
        val = 2 * d * (1 + r - np.exp(0.5 * (l * s_cd) ** 2)) * np.exp(0.5 * (l * s_cd) ** 2) + k * l * s_cd**2 * np.exp((l * s_cd) ** 2)
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  # noiseless adder + noise in growth rate
        s_l = par1['l_std'] * l
        val = 2*d*(1-exp_xy(-l, s_l, cd, s_cd))*exp_xy(l, s_l, cd, s_cd)
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = exp_xy(l, s_l, cd, s_cd) * (2*d-d*exp_xy(-l, s_l, cd, s_cd)-k*y_exp_xy(-l, s_l, cd, s_cd))
    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        s_l = par1['l_std'] * l
        val = 2*d*(1-exp_xy(-l, s_l, cd, s_cd))*exp_xy(l, s_l, cd, s_cd) * tay_e1(s_i)
    if par1['modeltype'] in [15, 16]:
        s_l = par1['l_std'] * l
        val = 2*d*par1['w_frac']*exp_xy(l, s_l, cd, s_cd)
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        s_l = par1['l_std'] * l
        val = 2 * d * (1 - exp_xy(-l, s_l, cd, s_cd)) * exp_xy(l, s_l, cd, s_cd)
    if par1['modeltype'] in [19, 20]:  # different model for mothers vs. daughters
        s_l = par1['l_std'] * l
        s_new = np.sqrt(par1['g1_std'] ** 2 + par1['g2_std'] ** 2) * par1['td']
        val = wb(par1, cell_no=1)*exp_xy(l, s_l, (par1['g1_delay']+par1['CD'])*par1['td'], s_new)
    return val


def vdvb_func(par1, s_i, s_cd):
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        k = par1['K'] / par1['td']  # per unit time
        d = k * cd
        a = 1 - 2 * np.exp(0.5 * (l * s_cd) ** 2) / (1 + r) + 2 * np.exp(2 * (l * s_cd) ** 2) / (1 + r) ** 2
        val = d ** 2 * (1 + r) * np.exp(0.5 * (l * s_cd) ** 2) * (np.exp(0.5 * (l * s_cd) ** 2) * ((1 + r) + 1 / (1 + r)) - 2) * (
            2 + 2 * a) / (2 - a)
    if par1['modeltype'] == 1:
        k = par1['K'] / par1['td']  # per unit time
        d = k * cd
        a = 1 - 2 * np.exp(0.5 * (l * s_cd) ** 2) / (1 + r) + 2 * np.exp(2 * (l * s_cd) ** 2) / (1 + r) ** 2
        temp = (3 * d**2*a + k**2*s_cd**2+8*d*k*l*s_cd**2*(np.exp(0.5*s_cd**2*l**2)-2*np.exp(2*l**2*s_cd**2)/(1+r))/
                (1+r)+2*k**2*s_cd**2*(-np.exp(0.5*s_cd**2*l**2)*(1+l**2*s_cd**2)+np.exp(2*s_cd**2*l**2)*
                                      (1+4*l**2*s_cd**2)/(1+r))/(1+r))/(2-a)
        val = np.exp(0.5*s_cd**2*l**2)*(1+r)*((temp+d**2+k*d*l*s_cd**2)*(1+r)*np.exp(0.5*s_cd**2*l**2)-2*(temp+d**2)+
                                               (temp+d**2-k*d*l*s_cd**2)*np.exp(0.5*s_cd**2*l**2)/(1+r))
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  # noiseless adder + noise in growth rate
        k = par1['K'] / par1['td']  # per unit time
        d = k * cd
        s_l = par1['l_std'] * l
        a = 1-2*exp_xy(-l, s_l, cd, s_cd)+2*exp_xy(-2*l, 2*s_l, cd, s_cd)
        val = (d**2*(2+2*a)/(2-a))*exp_xy(l, s_l, cd, s_cd)*(exp_xy(l, s_l, cd, s_cd)+exp_xy(-l, s_l, cd, s_cd)-2)
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        k = par1['K'] / par1['td']  # per unit time
        d = k * cd
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = exp_xy(l, s_l, cd, s_cd) * (wbwb_p(par1, s_i, s_cd) * (exp_xy(l, s_l, cd, s_cd) +
                                                                     exp_xy(-l, s_l, cd, s_cd) - 2) + d * k * (
                y_exp_xy(l, s_l, cd, s_cd) + y_exp_xy(-l, s_l, cd, s_cd) - 2*cd))

    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        k = par1['K'] / par1['td']  # per unit time
        d = k * cd
        s_l = par1['l_std'] * l
        a = 1 - 2 * exp_xy(-l, s_l, cd, s_cd) + 2 * exp_xy(-2 * l, 2 * s_l, cd, s_cd)
        val = (d ** 2 * (2 + 2 * a) / (2 - a)) * exp_xy(l, s_l, cd, s_cd) * (
            exp_xy(l, s_l, cd, s_cd) + exp_xy(-l, s_l, cd, s_cd) - 2) * tay_e1(s_i)**2
    if par1['modeltype'] in [15, 16]:
        k = par1['K'] / par1['td']  # per unit time
        d = k * cd
        f = par1['w_frac']
        s_l = par1['l_std'] * l
        val = f*exp_xy(l, s_l, cd, s_cd)*(wbwb_p(par1, s_i, s_cd)*(exp_xy(l, s_l, cd, s_cd)-1) +
                                          d*(k*y_exp_xy(l, s_l, cd, s_cd)-d))
    if par1['modeltype'] in [17, 18]:
        d = par1['delta']
        s_l = par1['l_std'] * l
        val = (wbwb_p(par1, s_i, s_cd) + d ** 2) * (
            exp_xy(l, s_l, cd, s_cd) + exp_xy(-l, s_l, cd, s_cd) - 2) * exp_xy(l, s_l, cd, s_cd)
    if par1['modeltype'] in [19, 20]:  # different model for mothers vs. daughters
        d = par1['delta']
        s_l = par1['l_std'] * l
        s_new = np.sqrt(par1['g1_std'] ** 2 + par1['g2_std'] ** 2) * par1['td']
        wbvb = exp_xy(l, s_l, par1['td']*(-par1['CD']+par1['g1_delay']), s_new)*\
               (d*(wb(par1, cell_no=1)+vb_m(par1, s_i, s_cd))+wbwb(par1, cell_no=1))/(2-exp_xy(l, s_l, par1['td']*(-par1['CD']+par1['g1_delay']), s_new))
        temp = 0.5*(wbwb(par1, cell_no=1)+wb(par1, cell_no=1)*d+vb_m(par1, s_i, s_cd)*d + wbvb)
        val = exp_xy(l, s_l, (par1['g1_delay']+par1['CD'])*par1['td'], s_new)*temp*\
            (exp_xy(l, s_l, (par1['g1_delay']+par1['CD'])*par1['td'], s_new) +
            exp_xy(l, s_l, (par1['g1_delay']-par1['CD'])*par1['td'], s_new) -
            2 * exp_xy(l, s_l, par1['g1_delay']*par1['td'], par1['g1_std']*par1['td']))
    return val


def vbvb_m(par1, s_i, s_cd):
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k*cd
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        a = 1-2*np.exp(0.5*(l*s_cd)**2)/(1+r)+2*np.exp(2*(l*s_cd)**2)/(1+r)**2
        val = 3*d**2*a/(2-a)+s_i**2
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  # noiseless adder + noise in growth rate
        s_l = par1['l_std'] * l
        a = 1-2*exp_xy(-l, s_l, cd, s_cd)+2*exp_xy(-2*l, 2*s_l, cd, s_cd)
        val = 3 * d ** 2 * a / (2 - a) + s_i ** 2
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        val = wbwb_p(par1, s_i, s_cd)+s_i**2
    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        s_l = par1['l_std'] * l
        a = 1 - 2 * exp_xy(-l, s_l, cd, s_cd) + 2 * exp_xy(-2 * l, 2 * s_l, cd, s_cd)
        val = 3 * d ** 2 * a / (2 - a) * tay_e2(s_i)
    if par1['modeltype'] in [15, 16]:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        val = wbwb_p(par1, s_i, s_cd)+s_i**2
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        val = wbwb_p(par1, s_i, s_cd) + (d*s_i) ** 2
    return val


def vb_m(par1, s_i, s_cd):
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k * cd
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        val = d
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  # noiseless adder + noise in growth rate
        val = d
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        val = d
    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        val = d * tay_e1(s_i)
    if par1['modeltype'] in [15, 16]:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        val = d
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        val = d
    if par1['modeltype'] in [19, 20]:
        s_l = par1['l_std'] * l
        val = wb(par1, cell_no=1) * exp_xy(l, s_l, par1['g1_delay']*par1['td'], par1['g1_std']*par1['td']) /\
                (2 - exp_xy(l, s_l, par1['g1_delay']*par1['td'], par1['g1_std']*par1['td']))
    return val


def vd_m(par1, s_i, s_cd):
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k * cd
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        val = 2*d*np.exp((l*s_cd)**2)
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  # noiseless adder + noise in growth rate
        s_l = par1['l_std'] * l
        val = 2*d*exp_xy(-l, s_l, cd, s_cd)*exp_xy(+l, s_l, cd, s_cd)
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = exp_xy(l, s_l, cd, s_cd)*(d*exp_xy(-l, s_l, cd, s_cd)+k*y_exp_xy(-l, s_l, cd, s_cd))
    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        s_l = par1['l_std'] * l
        val = 2 * d * exp_xy(-l, s_l, cd, s_cd) * exp_xy(+l, s_l, cd, s_cd) * tay_e1(s_i)
    if par1['modeltype'] in [15, 16]:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = 2*d*(1-par1['w_frac'])*exp_xy(l, s_l, cd, s_cd)
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        s_l = par1['l_std'] * l
        val = 2*d*exp_xy(-l, s_l, cd, s_cd)*exp_xy(+l, s_l, cd, s_cd)
    return val


def vdvb_m(par1, s_i, s_cd):
    cd = par1['CD'] * par1['td']
    td = par1['td']
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k * cd
    r = np.exp(l*cd)-1
    if par1['modeltype'] == 0:
        a = 1-2*np.exp(0.5*(l*s_cd)**2)/(1+r)+2*np.exp(2*(l*s_cd)**2)/(1+r)**2
        val = np.exp((l*s_cd)**2)*(2+2*a)/(2-a)*d**2
    if par1['modeltype'] == 4 or par1['modeltype'] == 9:  #noiseless adder + noise in growth rate
        s_l = par1['l_std'] * l
        a = 1-2*exp_xy(-l, s_l, cd, s_cd)+2*exp_xy(-2*l, 2*s_l, cd, s_cd)
        val = 2*d**2*(1+a)/(2-a)*exp_xy(l, s_l, cd, s_cd)*exp_xy(-l, s_l, cd, s_cd)
    if par1['modeltype'] == 5 or par1['modeltype'] == 10:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = exp_xy(l, s_l, cd, s_cd)*(wbwb_p(par1, s_i, s_cd)*exp_xy(-l, s_l, cd, s_cd)+d*k*y_exp_xy(-l, s_l, cd, s_cd))
    if par1['modeltype'] == 7:  # noiseless adder + noise in growth rate and multiplicative Start noise
        s_l = par1['l_std'] * l
        a = 1 - 2 * exp_xy(-l, s_l, cd, s_cd) + 2 * exp_xy(-2 * l, 2 * s_l, cd, s_cd)
        val = 2 * d ** 2 * (1 + a) / (2 - a) * exp_xy(l, s_l, cd, s_cd) * exp_xy(-l, s_l, cd, s_cd) * tay_e1(s_i) ** 2
    if par1['modeltype'] in [15, 16]:
        # noisy Whi5 adder + noise in growth rate, whi5 synth rate and additive Start noise.
        s_l = par1['l_std'] * l
        val = (1-par1['w_frac'])*(wbwb_p(par1, s_i, s_cd)+d**2)*exp_xy(l, s_l, cd, s_cd)
    if par1['modeltype'] in [17, 18]:
        del d
        d = par1['delta']
        s_l = par1['l_std'] * l
        val = (wbwb_p(par1, s_i, s_cd) + d ** 2)*exp_xy(l, s_l, cd, s_cd)*exp_xy(-l, s_l, cd, s_cd)
    return val


def slope_vbvd_func(par1,s_i,s_cd):
    f = (vdvb_func(par1, s_i, s_cd) - vd_func(par1, s_i, s_cd) * vb_func(par1, s_i, s_cd)) / (
        vbvb_func(par1, s_i, s_cd) - vb_func(par1, s_i, s_cd) ** 2)
    return f


def slope_vbvd_m(par1, s_i, s_cd):
    f = (vdvb_m(par1, s_i, s_cd) - vd_m(par1, s_i, s_cd) * vb_m(par1, s_i, s_cd)) / (
        vbvb_m(par1, s_i, s_cd) - vb_m(par1, s_i, s_cd) ** 2)
    return f

def slope_vbvd_p(par1, s_i, s_cd):
    f = (vdvb_m(par1, s_i, s_cd) - vd_m(par1, s_i, s_cd) * vb_m(par1, s_i, s_cd) + vdvb_func(par1, s_i, s_cd) -
         vd_func(par1, s_i, s_cd) * vb_func(par1, s_i, s_cd)) / (vbvb_m(par1, s_i, s_cd) - vb_m(par1, s_i, s_cd) ** 2 +
                                                                  vbvb_func(par1, s_i, s_cd) - vb_func(par1, s_i,
                                                                                                       s_cd) ** 2)
    return f


def slope_all_celltypes(par1, s_i, s_cd):
    f = np.zeros([3])
    f[0] = slope_vbvd_m(par1, s_i, s_cd)  # mothers
    f[1] = slope_vbvd_func(par1, s_i, s_cd)  # daughters
    f[2] = slope_vbvd_p(par1, s_i, s_cd)  # pop
    return f


def wb(par1, cell_no):
    val0 = par1['K'] * par1['CD']
    if par1['modeltype'] in [4, 9]:
        if cell_no == 0:  # mothers
            val1 = 2 * val0 * np.exp((np.log(2)*par1['g2_std'])**2/2)/2**par1['CD']
        elif cell_no == 1:  # daughters
            val1 = 2 * val0 * (1 - np.exp((np.log(2) * par1['g2_std']) ** 2 / 2) / 2 ** par1['CD'])
        elif cell_no == 2:
            val1 = val0
    if par1['modeltype'] in [13, 15, 16]:
        f = par1['w_frac']
        if cell_no == 0:  # mothers
            val1 = 2 * (1-f) * val0
        elif cell_no == 1:  # daughters
            val1 = 2 * f * val0
        elif cell_no == 2:
            val1 = val0
    if par1['modeltype'] in [19, 20]:  # different regulation for mothers and daughters!
        d = par1['delta']
        l = np.log(2)/par1['td']
        s_l = par1['l_std'] * l
        if cell_no == 0:  # mothers
            val1 = 2 * d * exp_xy(-l, s_l, par1['CD']*par1['td'], par1['g2_std']*par1['td'])
        elif cell_no == 1:  # daughters
            val1 = 2 * d * (1 - exp_xy(-l, s_l, par1['CD']*par1['td'], par1['g2_std']*par1['td']))
        elif cell_no == 2:
            val1 = d
    return val1


def wbwb(par1, cell_no):
    td = par1['td']
    cd = par1['CD'] * par1['td']  # units of time
    l = np.log(2) / td
    k = par1['K'] / par1['td']  # per unit time
    d = k*cd
    s_cd = par1['g2_std'] * td
    if par1['modeltype'] in [4, 9] and par1['l_std'] == 0:
        s_l = par1['l_std'] * l
        a = 1 - 2 * exp_xy(-l, s_l, cd, s_cd) + 2 * exp_xy(-2 * l, 2 * s_l, cd, s_cd)
        if cell_no == 0:
            val = d ** 2 * 3 * 2 * np.exp(2 * (l * s_cd) ** 2) / ((2 - a) * 2 ** (2 * cd / td))
        if cell_no == 1:
            val = 2 * d ** 2 * 3 * (1 - 2 * np.exp((l * s_cd) ** 2 / 2) / 2 ** (cd / td) + np.exp(2 * (l * s_cd) ** 2) /
                                    2 ** (2 * cd / td)) / (2 - a)
        if cell_no == 2:
            val = d ** 2 * 3 * a / (2 - a)
    if par1['modeltype'] in [13]:
        f = par1['w_frac']
        a = 1 - 2 * f + 2 * f ** 2
        if cell_no == 0:
            val = 2*d**2 * (1-f)**2 * (3)/(2-a)
        if cell_no == 1:
            val = 2*d ** 2 * f ** 2 * (3) / (2 - a)
        if cell_no == 2:
            val = d**2 * (3)*a/(2-a)
    if par1['modeltype'] in [15, 16]:
        f = par1['w_frac']
        a = 1 - 2 * f + 2 * f ** 2
        if cell_no == 0:
            val = 2*(1-f)**2 * (2 * d ** 2 + k ** 2 * (1 + par1['k_std'] ** 2) * (cd ** 2 + s_cd ** 2)) / (2 - a)
        if cell_no == 1:
            val = 2 * f**2 * (2 * d ** 2 + k ** 2 * (1 + par1['k_std'] ** 2) * (cd ** 2 + s_cd ** 2)) / (2 - a)
        if cell_no == 2:
            val = a*(2*d**2 + k**2*(1+par1['k_std']**2)*(cd**2+s_cd**2))/(2-a)
    if par1['modeltype'] in [19, 20]:  # different model for mothers vs. daughters
        s_l = par1['l_std'] * l
        s_new = np.sqrt(par1['g1_std'] ** 2 + par1['g2_std'] ** 2) * par1['td']
        del d
        d = par1['delta']
        a = 1 - 2 * exp_xy(-l, s_l, cd, s_cd) + 2 * exp_xy(-2 * l, 2 * s_l, cd, s_cd)
        s_i = par1['g1_thresh_std'] * d
        if cell_no == 0:  # mothers
            val = exp_xy(-2 * l, 2 * s_l, cd, s_cd) * (s_i ** 2 + 6 * d ** 2 / (2 - a))
        if cell_no == 1:  # daughters
            val = (1-2*exp_xy(-l, s_l, cd, s_cd)+exp_xy(-2*l, 2*s_l, cd, s_cd)) * (s_i ** 2 + 6 * d ** 2 / (2 - a))
        if cell_no == 2:
            val = d ** 2 * 3 * a / (2 - a)
    return val


def mean_vdvb(par1, cell_no):  # mean volume added in subsequent G1 phase for mothers or daughters
    val0 = par1['K'] * par1['CD']
    if par1['modeltype'] in [4, 9, 13]:
        val1 = val0 + wb(par1, cell_no)*(1-2**par1['CD']*np.exp((np.log(2)*par1['g2_std'])**2/2))
    return val1


def mean_vdvb1(par1, cell_no, next_cell_no):  # mean volume added in subsequent G1 phase for mothers or daughters
    val0 = par1['K'] * par1['CD']  # cell_no gives the initial cell, next_cell_no gives the type of next generation cell
    if par1['modeltype'] in [15, 16]:
        td = par1['td']
        l = np.log(2) / td
        s_l = par1['l_std'] * l
        s_cd = par1['g2_std'] * td
        cd = par1['CD'] * par1['td']  # units of time
        if next_cell_no == 0:  # mothers
            val1 = (1-par1['w_frac'])*val0 - par1['w_frac'] * wb(par1, cell_no)
        if next_cell_no == 1:  # daughters
            val1 = par1['w_frac'] * val0 + wb(par1, cell_no)*(1+par1['w_frac']-exp_xy(l, s_l, cd, s_cd))
    return val1


def std_vdvb(par1, cell_no):  # std deviation of volume added in subsequent G1 phase for mothers or daughters
    val0 = par1['K'] * par1['CD']
    val1 = val0**2 + 2*val0*wb(par1, cell_no)*(1-2**par1['CD']*np.exp((np.log(2)*par1['g2_std'])**2/2)) + \
            wbwb(par1, cell_no)*(1-2**(1+par1['CD'])*np.exp((np.log(2)*par1['g2_std'])**2/2)+2**(2*par1['CD']) *
                                  np.exp(2 * (np.log(2)*par1['g2_std'])**2))
    val1 = np.sqrt(val1 - mean_vdvb(par1, cell_no)**2)
    return val1


def std_vdvb1(par1, cell_no, next_cell_no):
    # std deviation of volume added in subsequent G1 phase for mothers or daughters
    # cell_no gives the initial cell, next_cell_no gives the type of next generation cell
    val0 = par1['K'] * par1['CD']
    if par1['modeltype'] in [15, 16]:
        td = par1['td']
        k = par1['K']
        cd=par1['CD']
        l = np.log(2) / td
        s_l = par1['l_std'] * l
        s_cd = par1['g2_std'] * td
        f = par1['w_frac']
        if next_cell_no == 0:  # mothers
            val1 = (1-f)**2*(wbwb(par1, cell_no)+2*wb(par1, cell_no)*val0+k**2*(1+par1['k_std']**2)*(val0**2+s_cd**2))+\
                   2 * par1['g1_thresh_std']**2+wbwb(par1,cell_no)-2*(1-f)*(wbwb(par1, cell_no)+wb(par1, cell_no)*val0)
        if next_cell_no == 1:  # daughters
            val1 = f**2*(wbwb(par1, cell_no)+2*wb(par1, cell_no)*val0 + k**2*(1+par1['k_std']**2)*(val0**2+s_cd**2)) + \
                   par1['g1_thresh_std']**2-2*f*(exp_xy(l, s_l, cd, s_cd)*wbwb(par1, cell_no)+
                                                 k*y_exp_xy(l, s_l, cd, s_cd)*wb(par1, cell_no) - wbwb(par1, cell_no) -
                                                 val0*wb(par1, cell_no))+(wbwb(par1, cell_no)+par1['g1_thresh_std']**2)*(exp_xy(2*l, 2*s_l, cd, s_cd)-2*exp_xy(l, s_l, cd, s_cd)+1)
        val1 = np.sqrt(val1 - mean_vdvb1(par1, cell_no, next_cell_no) ** 2)
    return val1


def fn_val_gen(par1, g1_std, g2_std):
    dims = [len(g1_std), len(g2_std)]
    slopes_func = np.empty(dims)
    for i in range(len(g1_std)):
        par1['g1_thresh_std'] = g1_std[i]
        for j in range(len(g2_std)):
            par1['g2_std'] = g2_std[j]
            slopes_func[i, j] = slope_vbvd_func(par1, g1_std[i], g2_std[j])
    return slopes_func


def heat_maps_test_fn(obs, slopes_func, g1_std, g2_std):
    model = 'Asymmetric division Whi5 noiseless adder'
    plt.figure(figsize=[18, 17])
    sns.heatmap(obs[:, :, 1]-slopes_func[:, :], xticklabels=np.around(g2_std, decimals=2), \
                     yticklabels=np.around(g1_std[::-1], decimals=2), annot=True)
    plt.xlabel('C+D timing noise $\sigma_{C+D}$', size=20)
    plt.ylabel('Start threshold noise $\sigma_{thresh}$', size=20)
    plt.title('Daughter theory simulation difference'+' '+model, size=20)
    plt.show()


def test_function(obs, par1):
    g1_std = np.linspace(0.0, 0.20, 21)
    g2_std = np.linspace(0.0, 0.20, 21)
    slopes_fn = fn_val_gen(par1, g1_std, g2_std)
    heat_maps_test_fn(obs, slopes_fn, g1_std, g2_std)
    plt.figure(figsize=[6,6])
    plt.plot(g2_std, obs[:, 5, 1], label='simulations')
    plt.plot(g2_std, slopes_fn[:, 5], label='theory')
    plt.title('$V_d$ $V_b$ slopes for $\sigma_{CD}$ ='+str(g2_std[20]))
    plt.xlabel('$\sigma_{thresh}')
    plt.legend()


def test_function_syst(obs, par1, g1_std, g2_std, vec):
    # This function allows us to systematically simulate in order to check the
    # agreement of different theoretical expressions with the simulations. We can see that this appears to be perfectly
    # fine for the variation with respect to
    labels = ['$V_d$ vs. $V_b$ slope', '$<V_b>$', '$<Vb^2>$', '$<V_d>$', '$<V_dV_b>$']
    values = range(len(vec))
    cmap = plt.get_cmap('cool')
    cnorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    fig1 = plt.figure(figsize=[15, 18])
    g2_std_m = np.linspace(g2_std[0], g2_std[-1], 100)
    for j in range(obs.shape[2] - 1):
        plt.subplot(3, 2, j + 1)
        for i in range(len(vec)):
            colorval = scalarmap.to_rgba(values[i])
            vals = np.empty((len(labels), len(g2_std_m)))
            vals[1, :] = vb_func(par1, g1_std[vec[i]], g2_std_m)
            vals[2, :] = vbvb_func(par1, g1_std[vec[i]], g2_std_m)
            vals[3, :] = vd_func(par1, g1_std[vec[i]], g2_std_m)
            vals[4, :] = vdvb_func(par1, g1_std[vec[i]], g2_std_m)
            vals[0, :] = (vals[4, :]-vals[3, :] * vals[1, :])/(vals[2, :]-vals[1, :]**2)
            plt.plot(g2_std_m, vals[j, :], label='theory $\sigma_{G1}=$ '+str(g1_std[vec[i]]), linewidth=4.0, color=colorval)
            # plt.scatter(g2_std, obs[vec[i], :, j, 1], marker='o', s=8.0, label='sim $\sigma_{G1}=$ '+str(g1_std[vec[i]]))
            plt.scatter(g2_std, obs[vec[i], :, j, 1], marker='o', s=12.0, color=colorval)
            plt.title(labels[j] + ' daughter ' + models[par1['modeltype']])
            plt.xlabel('$\sigma_{G2}$')
            plt.legend(loc=4)
    fig2 = plt.figure(figsize=[15, 18])
    if par1['mothervals']:
        for j in range(obs.shape[2] - 1):
            plt.subplot(3, 2, j + 1)
            for i in range(len(vec)):
                colorval = scalarmap.to_rgba(values[i])
                vals = np.empty((len(labels), len(g2_std_m)))
                vals[1, :] = vb_m(par1, g1_std[vec[i]], g2_std_m)
                vals[2, :] = vbvb_m(par1, g1_std[vec[i]], g2_std_m)
                vals[3, :] = vd_m(par1, g1_std[vec[i]], g2_std_m)
                vals[4, :] = vdvb_m(par1, g1_std[vec[i]], g2_std_m)
                vals[0, :] = (vals[4, :]-vals[3, :] * vals[1, :])/(vals[2, :]-vals[1, :]**2)
                plt.plot(g2_std_m, vals[j, :], label='theory $\sigma_{G1}=$ ' + str(g1_std[vec[i]]), linewidth=4.0, color=colorval)
                # plt.scatter(g2_std, obs[vec[i], :, j, 0], marker='o', s=8.0, label='sim $\sigma_{G1}=$ ' + str(g1_std[vec[i]]))
                plt.scatter(g2_std, obs[vec[i], :, j, 0], marker='o', s=12.0, color=colorval)
                plt.title(labels[j] + ' mother '+models[par1['modeltype']])
                plt.xlabel('$\sigma_{G2}$')
                plt.legend(loc=4)
    return fig1, fig2  # returns daughters first, then mothers.


def test_slopes(obs):

    new = obs[::-1,:,:,:]
    g1_std = np.linspace(0.0, 0.23, 24)
    g2_std = np.linspace(0.0, 0.23, 24)
    inferred_slope = (new[:, :, 4, :] - new[:, :, 3, :] * new[:, :, 1, :]) / (
        new[:, :, 2, :] - new[:, :, 1, :] ** 2)
    heat_maps_mother_daughter(new[:,:,0,:], g1_std, g2_std, 'measured slope')
    heat_maps_mother_daughter(inferred_slope, g1_std, g2_std, 'inferred slope')


def plot_systematically(obs, g1_std, g2_std, par1):
    # Select a color map
    values = range(len(g1_std))
    cmap = cm = plt.get_cmap('gnuplot2')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    celltype = ['Mothers','Daughters']
    fig = plt.figure(figsize=[15, 7])
    k = par1['K']
    cd = par1['CD']
    model = [2 * k*cd * np.exp(0.5 * (np.log(2) * g2_std/par1['td']) ** 2) / 2 ** cd,
             2 * k*cd * (1 - np.exp(0.5 * (np.log(2) * g2_std/par1['td']) ** 2) / 2 ** cd)]
    for j in range(2):
        ax=plt.subplot(1,2,j+1)
        for i in range(obs.shape[0]):
            colorVal = scalarMap.to_rgba(values[i])
            lab='$\sigma_{G1_{thr}}=$ ' + str(g1_std[i])
            plt.plot(g2_std,obs[i,:,5,j],color=colorVal,label=lab)
        plt.plot(g2_std,model[j],label='theory',color='g')
        plt.xlabel('$\sigma_{G2}$',size=20)
        plt.ylabel('$<w_b>$',size=20)
        plt.title(celltype[j]+' Whi5 at birth for variable noise',size=20)
        plt.legend()
    return fig


def plot_vb_systematically(obs,g1_std,g2_std,par1):
    # Select a color map
    values=range(len(g1_std))
    cmap = cm = plt.get_cmap('cool')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    celltype=['Mothers','Daughters']
    fig = plt.figure(figsize=[15, 7])
    k = par1['K']
    cd = par1['CD']
    model = [k*cd*np.ones(g2_std.shape),
             k*cd * (2**(cd/par1['td'])*np.exp(0.5 * (np.log(2) * g2_std/par1['td']) ** 2)-1)]
    for j in range(2):
        ax=plt.subplot(1,2,j+1)
        for i in range(obs.shape[0]):
            colorVal = scalarMap.to_rgba(values[i])
            lab='$\sigma_{G1_{thr}}=$ ' + str(g1_std[i])
            plt.plot(g2_std,obs[i,:,1,j],color=colorVal,label=lab)
        plt.plot(g2_std,model[j],label='theory')
        plt.xlabel('$\sigma_{G2}$',size=20)
        plt.ylabel('$<V_b>$',size=20)
        plt.title(celltype[j]+' $V_b$ for variable noise',size=20)
        plt.legend()
    return fig


def test_deviation(obs,g2_std,par1):  # useful for looking at what the form is of the difference between the predicted
    # and observed values.
    celltype=['Mothers','Daughters']
    sims=np.mean(obs, axis=0)
    fig = plt.figure(figsize=[15, 7])
    k=par1['K']
    cd=par1['CD']
    model = [2 * k*cd * np.exp(0.5 * (np.log(2) * g2_std/par1['td']) ** 2) / 2 ** cd,
             2 * k*cd * (1 - np.exp(0.5 * (np.log(2) * g2_std/par1['td']) ** 2) / 2 ** cd)]
    for j in range(2):
        ax=plt.subplot(1, 2, j+1)
        plt.plot(g2_std,model[j]/sims[:,5, j])
        plt.xlabel('$\sigma_{G2}$',size=20)
        plt.ylabel('$<w_{b,mod}>/<w_{b,sim}>$',size=20)
        plt.title(celltype[j]+' Whi5 at birth for variable noise',size=20)
    return fig


def test_population_av(obs,g2_std,par1):  # useful for looking at what the form is of the difference between the predicted
    # and observed values.
    celltype=['Mothers','Daughters']
    sims=np.mean(obs, axis=3)
    sims=np.mean(sims, axis=0)
    fig = plt.figure(figsize=[7, 7])
    k=par1['K']
    cd=par1['CD']
    model = k*cd*np.ones(g2_std.shape)
    plt.plot(g2_std,model,label='theory')
    plt.plot(g2_std, sims[:,5], label='theory')
    plt.xlabel('$\sigma_{G2}$',size=20)
    plt.ylabel('$<w_b>$',size=20)
    plt.title('Population average'+' Whi5 at birth for variable noise',size=20)
    return fig
########################################################################################################################


def func(par1):
    tic = time.clock()
    g1_std = np.linspace(0.0, 0.10, 10)
    g2_std = np.linspace(0.0, 0.10, 10)
    cd = np.linspace(0.5, 1.0, 11)
    dims = [len(g1_std), len(g2_std), 2, len(cd)]
    slopes = np.empty(dims)
    par1['initiator'] = 0  # dilution model
    for k in range(len(cd)):
        par1['CD'] = cd[k]
        for i in range(len(g1_std)):
            par1['g1_thresh_std'] = g1_std[-1 - i]
            for j in range(len(g2_std)):
                par1['g2_std'] = g2_std[j]
                c, num_cells, tvec = discr_time(par1)
                vb = [obj.vb for obj in c[1000:] if obj.isdaughter]
                vd = [obj.vd for obj in c[1000:] if obj.isdaughter]
                vals = scipy.stats.linregress(vb, vd)
                slopes[i, j, 0, k] = vals[0]
                del vals, vb, vd
                vb = [obj.vb for obj in c[1000:] if not obj.isdaughter]
                vd = [obj.vd for obj in c[1000:] if not obj.isdaughter]
                vals = scipy.stats.linregress(vb, vd)
                slopes[i, j, 1, k] = vals[0]
        print('time taken: ', time.clock() - tic)
    return slopes


def var_heat_maps(slopes, g1_std, g2_std,cd):
    #  Takes slopes as input and plots correlations between volume at birth for mother and daughters, labelled by the
    #  asymmetry ratio used to generate that data
    font = {'family': 'normal', 'weight': 'bold', 'size': 12}
    plt.rc('font', **font)
    celltype = ['Daughter', 'Mother']
    r = np.round(np.exp(np.log(2)*cd/par['td'])-1,3)
    for j in range(slopes.shape[2]):
        for i in range(slopes.shape[3]):
            plt.figure(figsize=[11, 10])
            sns.heatmap(slopes[:, :, j, i], xticklabels=np.around(g2_std, decimals=2), \
                        yticklabels=np.around(g1_std[::-1], decimals=2), annot=True)
            plt.xlabel('C+D timing noise $\sigma_{C+D}$', size=20)
            plt.ylabel('Start threshold noise $\sigma_{thresh}$', size=20)
            plt.title('Asymmetric dilution '+celltype[j]+' $r=$'+str(r[i])+' $t_{bud}$ ='+str(cd[i]), size=20)
            plt.show()


def single_par_test(par1, val):
    # This function returns a list containing the slope between birth and division for a single input parameter
    celltype = [' Mothers', ' Daughters']
    c = discr_gen(par1)
    x = np.asarray([obj.vb for obj in c[1000:] if obj.isdaughter == val])
    y = np.asarray([obj.vd for obj in c[1000:] if obj.isdaughter == val])
    num_bins = 20
    l = 3.5
    fig=plt.figure(figsize=[7, 7], facecolor='white')
    ax=plt.subplot(1,1,1)
    ax.set_axis_bgcolor("w")
    plt.hexbin(x, y, cmap="Purples", gridsize=60)
    tit = plt.title(models[par1['modeltype']][:-2] + celltype[val], size=16, weight="bold")
    plt.xlabel("$V_b$", size=16, weight="bold")
    plt.ylabel("$V_d$", size=16, weight="bold")
    plt.ylim(ymin=np.mean(y) - l * np.std(y), ymax=np.mean(y) + l * np.std(y))
    plt.xlim(xmin=np.mean(x) - l * np.std(x), xmax=np.mean(x) + l * np.std(x))
    bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x, np.arange(len(x)),
                                                                   statistic='mean', bins=num_bins,
                                                                   range=None)
    bin_posns = list([])
    y_av = list([])
    y_sem = list([])
    bin_err = list([])
    for j in range(num_bins):
        bin_no = binnumber == j + 1
        if np.sum(bin_no) > 20:  # we don't want to be skewed by having too few data points in a fit
            y_av.append(np.mean(y[np.nonzero(bin_no)]))
            y_sem.append(np.std(y[np.nonzero(bin_no)]) / np.sqrt(np.sum(bin_no)))
            # y_sem.append(np.std(y[np.nonzero(bin_no)]))
            bin_posns.append((bin_edges[j] + bin_edges[j + 1]) / 2)
            bin_err.append((bin_edges[j + 1] - bin_edges[j]) / 2)
    y_av = np.asarray(y_av)
    y_sem = np.asarray(y_sem)
    bin_posns = np.asarray(bin_posns)
    plt.errorbar(bin_posns, y_av, yerr=y_sem, label="binned means", ls="none", color="r")
    xvals = np.linspace(np.mean(x) - 2.5 * np.std(x), np.mean(x) + 2.5 * np.std(x), 2)
    # Generate values for plotting fits
    vals = stats.linregress(x, y)
    plt.plot(xvals, xvals * vals[0] + vals[1], 'b-', label="Regression slope =" + str(np.round(vals[0], 2)))
    plt.legend()
    return fig


def dplot(x, y, st):  # a useful generic plotting piece of code
    vals = scipy.stats.linregress(x,y)
    xmax = np.mean(x) + 2.5 * np.std(x)
    xmin = np.mean(x) - 2.5 * np.std(x)
    ymin = np.mean(y) - 2.5 * np.std(y)
    ymax = np.mean(y) + 2.5 * np.std(y)
    xv = np.array([xmin, xmax])
    plt.figure(figsize=[10, 10])
    plt.hexbin(x, y, cmap="Purples", gridsize=50)
    plt.plot(xv, vals[0] * xv + vals[1], label='slope= ' + '%.3f' % (vals[0]) + '$\pm$''%.3f' % (vals[4]))
    plt.title(st, size=30, y=1.04)
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.legend(fontsize='x-large')
    plt.xlabel('$V_b$', size=30, weight='bold')
    plt.ylabel('$V_d$', size=30, weight='bold')
    plt.show()
    return vals


def parameter_variation_asymmetric(par1):  # here we only look at the dilution growth policy, and are
    # saving the result for both mothers and daughters
    tic = time.clock()
    g1_std = np.linspace(0.0, 0.20, 21)
    g2_std = np.linspace(0.0, 0.20, 21)
    dims = [len(g1_std), len(g2_std), 2]
    slopes = np.empty(dims)
    pcc = np.empty(dims)
    error = np.empty(dims)
    var = np.empty([len(g1_std), len(g2_std), 2, 4])
    for i in range(len(g1_std)):
        par1['g1_thresh_std'] = g1_std[-1 - i]
        for j in range(len(g2_std)):
            par1['g2_std'] = g2_std[j]
            c, num_cells, tvec = discr_time(par1)
            for k in range(2):
                vb = [obj.vb for obj in c[1000:] if obj.isdaughter == k]  # mothers are at 0, then daughters are at 1.
                vd = [obj.vd for obj in c[1000:] if obj.isdaughter == k]
                vals = scipy.stats.linregress(vb, vd)
                slopes[i, j, k] = vals[0]
                del vals
                vals = scipy.stats.pearsonr(vb, vd)
                pcc[i, j, k] = vals[0]
                error[i, j, k] = vals[1]
                del vals
                var[i, j, k, 0] = np.mean(vb)
                var[i, j, k, 1] = np.std(vb)
                var[i, j, k, 2] = np.mean(vd)
                var[i, j, k, 3] = np.mean(np.asarray(vd)*np.asarray(vb))
        print('time taken: ', time.clock() - tic)
    obs = [slopes, pcc, error, var]
    return obs


def nice_plot(ax_handle, tkw):
    # assumes that the current figure is the one that ax refers to
    ax_handle.set_facecolor('white')
    ax_handle.tick_params(axis='x', colors='black', **tkw)
    ax_handle.tick_params(axis='y', colors='black', **tkw)
    tmp = [ax_handle.get_ylim(), ax_handle.get_xlim()]
    plt.axhline(tmp[0][0], color='black')
    plt.axvline(tmp[1][0], color='black')
    plt.axhline(tmp[0][1], color='black')
    plt.axvline(tmp[1][1], color='black')
    plt.xticks(size=18)
    plt.yticks(size=18)
    return ax_handle


def plot_vals_array(data,temp,temp1):
    #  Variation of plt_all_conditions to look at two variables: x and y.
    # medium is rows, x is first column, y is second column. Temp gives titles.
    fig = plt.figure(figsize=[15, 15])
    l = 3  # number of standard deviations around mean value which will be plotted.
    vals = np.zeros([len(data), 5])
    num_bins = 20
    font = {'family': 'normal', 'weight': 'bold', 'size': 22}
    plt.rc('font', **font)
    for k in range(len(data)):
        ax = plt.subplot(1, len(data), k+1)
        ax.set_axis_bgcolor('white')
        x = data[k][:,0]
        y = data[k][:,1]
        plt.hexbin(x, y, cmap="Purples")
        # plt.plot(x, y, '.')
        plt.title(temp[k], size=16, weight="bold")
        plt.xlabel(temp1[0], size=16, weight="bold")
        plt.ylabel(temp1[1], size=16, weight="bold")
        xmin, xmax, ymin, ymax = np.mean(x) - l * np.std(x), np.mean(x) + l * np.std(x), np.mean(y) - l * np.std(y), np.mean(y) + l * np.std(y)
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.xlim(xmin=xmin, xmax=xmax)
        bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x, np.arange(len(x)), statistic='mean',
                                                                       bins=num_bins, range=None)
        bin_posns = list([])
        y_av = list([])
        y_sem = list([])
        bin_err = list([])
        for j in range(num_bins):
            bin_no = binnumber == j + 1
            if np.sum(bin_no) > 20:  # we don't want to be skewed by having too few data points in a fit
                y_av.append(np.mean(y[np.nonzero(bin_no)]))
                y_sem.append(np.std(y[np.nonzero(bin_no)]) / np.sqrt(np.sum(bin_no)))
                bin_posns.append((bin_edges[j] + bin_edges[j + 1]) / 2)
                bin_err.append((bin_edges[j + 1] - bin_edges[j]) / 2)
        y_av = np.asarray(y_av)
        y_sem = np.asarray(y_sem)
        bin_posns = np.asarray(bin_posns)
        bin_err = np.asarray(bin_err)
        plt.errorbar(bin_posns, y_av, yerr=y_sem, label="binned means", ls="none", color="r")
        vals[k,:]=scipy.stats.linregress(x,y)
        xvals = np.linspace(xmin, xmax, 100)
        # plt.plot(xvals, xvals*vals[k,0]+vals[k,1], label='LR slope={0}'.format(np.round(vals[k,0],2)))
        plt.plot(xvals, xvals*vals[k,0]+vals[k,1], label='$y={0}x+{1}$'.format(np.round(vals[k, 0], 3), np.round(vals[k,1],3)))
        plt.legend(loc=2)

        # fontsize of the axes title
        del x, y, bin_means, bin_edges, binnumber, y_av, y_sem, k, j
    plt.suptitle(temp1[0]+' vs. ' + temp1[1])
    return fig
