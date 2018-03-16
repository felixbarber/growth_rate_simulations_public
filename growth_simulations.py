import numpy as np
import scipy
import scipy.optimize
import weakref
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import stats
import pandas as pd

par = dict([('num_s', 50), ('vd', 1.0), ('vm', 1.0), ('std_v', 0.1)])

# modeltype 0 is the model from cerulus et al., with an asymmetrically dividing, non-budding growth morphology defined
# by r
# modeltype 1 is a simple adder model, with an asymmetrically dividing, non-budding growth morphology defined by r


def fn(L, tm, td):
    return np.exp(-L * td) + np.exp(-L * tm) - 1.0


def fn_1(L, tm, td):
    return (2*np.exp(L*tm)-3)/(2*(np.exp(L*tm)-1))-1/(2*(np.exp(L*td)-1))


class Cell(object):
    cellCount = 0  # total number of cells

    def __init__(self, birth_parameters):  # birth parameters = [tb, celltype, parent, vb, parent_current]
        self.exists = True
        # these are present for all cells
        self.tb = birth_parameters[0]
        self.celltype = birth_parameters[1]  # 0 is mother, 1 is daughter
        self.parent = birth_parameters[2]
        self.vb = birth_parameters[3]
        self.parent_current = birth_parameters[4]  # only used for daughter cells

    def grow_g1(self, par1):  # par1 contains all details for the specified cell
        # growth function for the first generation of a simulation

        if par1['modeltype'] == 0:
            # constant growth rate, with noise in division times and epigenetic inheritance of td times
            # as in stoch, epig from Cerulus et al. 2016
            if self.celltype == 0:  # mothers
                # print 'A'
                td_out_zscore = np.random.normal(0, par1['trans_std_mm'], size=1)[0]
                # print td_out_zscore
                self.t_grow = td_out_zscore * par1['td_std'][self.celltype] + par1['td'][self.celltype]
                # print par1['td'][self.celltype]
                self.td_zscore = (self.t_grow - par1['td'][self.celltype]) / par1['td_std'][self.celltype]
                self.td_out_zscores = []
                # output z score for mother mother
                # print par1['A_mm'], self.td_zscore, par1['trans_std_mm'], self.t_grow
                self.td_out_zscores.append(
                    np.random.normal(par1['A_mm'] * self.td_zscore, par1['trans_std_mm'], size=1)[0])
                # output z score for mother daughter
                self.td_out_zscores.append(
                    np.random.normal(par1['A_md'] * self.td_zscore, par1['trans_std_md'], size=1)[0])
            else:  # daughters
                # print 'B'
                td_out_zscore = np.random.normal(0, par1['trans_std_md'], size=1)[0]
                self.t_grow = td_out_zscore * par1['td_std'][self.celltype] + par1['td'][self.celltype]
                # print par1['td'][self.celltype]
                self.td_zscore = (self.t_grow - par1['td'][self.celltype]) / par1['td_std'][self.celltype]
                self.td_out_zscores = [np.random.normal(0.0, 1.0, size=1)[0]]  # only one output z-score in this case
            # we effectively seed the first generation as though born from parents with a z-score of zero
            #     print par1['A_mm'], self.td_zscore, par1['trans_std_mm'], self.t_grow
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
        elif par1['modeltype'] == 1:  # simple adder model
            temp_zscore = np.random.normal(0.0, 1.0, size=1)[0]
            if par1['lambda_std'] is None:
                self.t_grow = np.log(1 + par1['delta'] / self.vb) / par1['lambda'] + temp_zscore*par1['td_std'][self.celltype]
                self.t_div = self.tb + self.t_grow
                self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            else:
                temp = np.random.normal(0.0, 1.0, size=1)[0]
                self.gr = par1['lambda']*(1+par1['lambda_std']*temp)
                self.t_grow = np.log(1 + par1['delta'] / self.vb) / self.gr + temp_zscore * par1['td_std'][
                    self.celltype]
                self.t_div = self.tb + self.t_grow
                self.vd = self.vb * np.exp(self.gr * self.t_grow)
        elif par1['modeltype'] == 2:  # alpha tunable model from Lin and Amir
            temp_zscore = np.random.normal(0.0, 1.0, size=1)[0]
            if par1['lambda_std'] is None:
                self.t_grow = np.log((2*par1['alpha']*par1['delta']+2*(1.0-par1['alpha'])*self.vb)/self.vb)/\
                              par1['lambda']+par1['td_std'][self.celltype]*temp_zscore
                self.t_div = self.tb + self.t_grow
                self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            else:
                temp = np.random.normal(0.0, 1.0, size=1)[0]
                self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp)
                self.t_grow = np.log(
                    (2 * par1['alpha'] * par1['delta'] + 2 * (1.0 - par1['alpha']) * self.vb) / self.vb) / \
                              self.gr + par1['td_std'][self.celltype] * temp_zscore
                self.t_div = self.tb + self.t_grow
                self.vd = self.vb * np.exp(self.gr * self.t_grow)
            # print self.t_grow
        # self.t_div = self.tb + np.amax([self.t_grow, 0.0])


        Cell.cellCount += 1

    def grow(self, par1):  # par1 contains all details for the specified cell

        if par1['modeltype'] == 0:
            # constant growth rate, with noise in division times and epigenetic inheritance of td times
            # as in stoch, epig from Cerulus et al. 2016

            # there are 4 cases: one for each of parent = mother or daughter, one for each of current cell = mother or
            # daughter

            if self.celltype == 0:  # if the current cell is a mother
                self.t_grow = self.parent.td_out_zscores[0] * par1['td_std'][self.celltype] \
                              + par1['td'][self.celltype]  # the first z score corresponds to that for the next gen
                # calculating z_scores for the next generation
                self.td_zscore = (self.t_grow - par1['td'][self.celltype]) / par1['td_std'][self.celltype]
                self.td_out_zscores = []
                # output z score for mother mother
                self.td_out_zscores.append(
                    np.random.normal(par1['A_mm'] * self.td_zscore, par1['trans_std_mm'], size=1)[0])
                # output z score for mother daughter
                self.td_out_zscores.append(
                    np.random.normal(par1['A_md'] * self.td_zscore, par1['trans_std_md'], size=1)[0])
            else:  # if the current cell is a daughter
                self.t_grow = self.parent_current.td_out_zscores[1] * par1['td_std'][self.celltype] \
                              + par1['td'][self.celltype]  # the second z score corresponds to the current daughter
                self.td_out_zscores = [np.random.normal(0.0,1.0,size=1)[0]]  # only one output z-score in this case
            self.t_div = self.tb + np.amax([self.t_grow, 0.0])
            self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
        elif par1['modeltype'] == 1:  # simple adder model
            temp_zscore = np.random.normal(0.0, 1.0, size=1)[0]
            if par1['lambda_std'] is None:
                self.t_grow = np.log(1 + par1['delta'] / self.vb) / par1['lambda'] + temp_zscore * par1['td_std'][
                    self.celltype]
                self.t_div = self.tb + np.amax([self.t_grow, 0.0])
                self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            else:
                temp = np.random.normal(0.0, 1.0, size=1)[0]
                self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp)
                self.t_grow = np.log(1 + par1['delta'] / self.vb) / self.gr + temp_zscore * par1['td_std'][
                    self.celltype]
                self.t_div = self.tb + np.amax([self.t_grow, 0.0])
                self.vd = self.vb * np.exp(self.gr * self.t_grow)
            # print self.t_grow
        elif par1['modeltype'] == 2:  # alpha tunable model from Lin and Amir
            temp_zscore = np.random.normal(0.0, 1.0, size=1)[0]
            if par1['lambda_std'] is None:
                self.t_grow = np.log((2*par1['alpha']*par1['delta']+2*(1.0-par1['alpha'])*self.vb)/self.vb)/\
                              par1['lambda']+par1['td_std'][self.celltype]*temp_zscore
                self.t_div = self.tb + max(self.t_grow, 0.0)
                self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            else:
                temp = np.random.normal(0.0, 1.0, size=1)[0]
                self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp)
                self.t_grow = np.log(
                    (2 * par1['alpha'] * par1['delta'] + 2 * (1.0 - par1['alpha']) * self.vb) / self.vb) / \
                              self.gr + par1['td_std'][self.celltype] * temp_zscore
                self.t_div = self.tb + max(self.t_grow, 0.0)
                self.vd = self.vb * np.exp(self.gr * self.t_grow)
        Cell.cellCount += 1


    def size(self, par1, t):  # this evaluates the volume of this cell at a particular point in time
        # if par1['modeltype']==1:
        if par1['lambda_std'] is None:
            temp = par1['lambda']
        else:
            temp = self.gr
        temp1 = self.vb*np.exp(temp*(t-self.tb))
        return temp1


def starting_popn(par1):
    # To clarify we first set the initial condition for the simulation.
    l = par1['lambda']
    if par1['modeltype'] == 0:  # in this case, there is no size regulation so av cell size is set by the initial
        # condition. Note that by contrast with model 1, we have the same relative scaling between vd and vm to begin.
        vd = par1['r']*par1['v_init']
        vm = par1['v_init']
    elif par1['modeltype'] == 1:  # set the average size distributions to begin with here.
        vm = 2.0*par1['delta']/(1+par1['r'])
        vd = 2*par1['r']*par1['delta']/(1+par1['r'])
    elif par1['modeltype'] == 2:  # set the average size distributions to begin with here.
        vm = par1['delta']/(1+par1['r'])
        vd = par1['r']*par1['delta']/(1+par1['r'])

    v_init_d = np.random.normal(loc=vd, scale=par['std_v'] * vd, size=par['num_s'])
    v_init_m = np.random.normal(loc=vm, scale=par['std_v'] * vm, size=par['num_s'])

    t_div = np.zeros(par['num_s'])

    for i0 in range(len(t_div)):
        temp = 2.0
        while temp > 1.0:
            temp = np.random.exponential(scale=l, size=1)[0]
            # we expect that these cells have been caught at
            # some exponentially distributed point of progression through their cell cycles.
        t_div[i0] = temp
        c = []

    for i in range(par['num_s']):  # instantiating our initial population of cells. These cells do not have mothers.
        # print i
        # Daughters
        temp = [0.0, 1, None, v_init_d[i], None]
        c.append(Cell(temp))
        c[-1].grow_g1(par1)
        c[-1].t_div *= (1.0-t_div[i])
        c[-1].tb = c[-1].t_div - c[-1].t_grow
        c[-1].exists=True
        # Mothers
        temp = [0.0, 0, None, v_init_m[i], None]
        c.append(Cell(temp))
        c[-1].grow_g1(par1)
        c[-1].t_div *= (1.0 - t_div[i])
        c[-1].tb = c[-1].t_div - c[-1].t_grow
        c[-1].exists = True
        # defined in this manner all starting cells have been born at time less than or equal to 0.
    del v_init_d, v_init_m
    return c


    # how to set growth rate in such a way that the size distribution doesn't diverge? Implies that growth rate is tuned?
    # Seems unrealistic. May be instead that growth rate is modulated.


def next_gen(index, f, t, par1):
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    # frac = max((f[index].vd-f[index].vi)/f[index].vd, 0.0)
    frac1 = par1['r']/(1+par1['r'])

    # add new cell for new cycle of mother cell.
    temp = [t, 0, weakref.proxy(f[index]), (1-frac1) * f[index].vd, None]
    f.append(Cell(temp))
    f[-1].grow(par1)  # grow newborn cell
    f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
    f[index].exists = False  # track that this cell no longer "exists".
    # daughters
    temp = [t, 1, weakref.proxy(f[index]), frac1*f[index].vd, weakref.proxy(f[-1])]
    f.append(Cell(temp))
    # Produce a new cell based on the previous one and append it to the end of the list.
    f[-1].grow(par1)  # grow newborn cell
    f[index].daughter = weakref.proxy(f[-1])  # Update the mother cell to show this cell as a daughter.

    return f


def next_gen_troubleshooter(index, f, t, par1):
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    # frac = max((f[index].vd-f[index].vi)/f[index].vd, 0.0)
    frac1 = par1['r']/(1+par1['r'])

    # add new cell for new cycle of mother cell.
    temp = [t, 0, weakref.proxy(f[index]), (1-frac1) * f[index].vd, None]
    f.append(Cell(temp))
    f[-1].grow(par1)  # grow newborn cell
    f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
    f[index].exists = False  # track that this cell no longer "exists".
    # daughters
    temp = [t, 1, weakref.proxy(f[index]), frac1*f[index].vd, weakref.proxy(f[-1])]
    temp1=Cell(temp)
    # Produce a new cell based on the previous one and append it to the end of the list.
    temp1.grow(par1)  # grow newborn cell
    f[index].daughter = weakref.proxy(temp1)  # Update the mother cell to show this cell as a daughter.

    return f, temp1


def discr_time_1(par1, starting_pop):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a set of parameters par1 and a starting population of cells.
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'] * np.log(2)/par1['lambda'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    num_existent_cells = np.zeros(tvec.shape)
    num_div_d = np.zeros(tvec.shape)  # keep track of the number of divisions from mother cells
    num_div_m = np.zeros(tvec.shape)  # keep track of the number of divisions from daughter cells
    av_v = np.zeros(tvec.shape)
    std_v = np.zeros(tvec.shape)
    vol = np.zeros(tvec.shape)
    # Define lists which will keep track of the time step in which each cell divides.
    div_times = []
    for i in range(nstep + 1):
        div_times.append([])
    # Now we go through our starting population and determine at which time step they will divide (cells with a division
    # of all cells and store that.
    c = starting_pop[:]
    num_cells[0] = len(c)
    num_existent_cells[0] = len(c)
    for i in range(len(c)):
        if c[i].t_div < np.amax(tvec):  # ensures we never consider cells which would go into the (nstep+2)th bin, since
            # there is none in this setup
            td_ind = np.searchsorted(tvec, np.array(c[i].t_div), side='left', sorter=None)
            # left means that e.g. (n-1)*dt<x<=n*dt goes to the nth bin.
            # nstep*dt<x means that the index is (nstep+1).
            # print td_ind, i
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
            if c[index].celltype:
                num_div_d[i] += 1
            elif not c[index].celltype:
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
        vol[i] = np.sum([obj.size(par1, tvec[i]) for obj in c if obj.exists])  # cumulative size of the cells at each
        # timestep
        num_cells[i] = len(c)
        temp_val = [obj.vb for obj in c if obj.exists]
        av_v[i] = np.mean(temp_val)
        std_v[i] = np.std(temp_val)
        num_existent_cells[i] = len(temp_val)
    obs = [num_cells, tvec, num_div_d, num_div_m, num_existent_cells, av_v, std_v, vol]
    return c, obs


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


def heat_maps(obs, labels, x_std, y_std):  # assumes obs has y then x
    font = {'family': 'normal', 'weight': 'bold', 'size': 12}
    plt.rc('font', **font)
    model = ['dilution symmetric', 'initiator symmetric']
    # plots = [0, 1, 3, 4]
    figs=[]
    for i in range(obs.shape[0]):
        figs.append(plt.figure(figsize=[11, 10]))
        sns.heatmap(obs[i, ::-1, :], xticklabels=np.around(x_std, decimals=2), \
                         yticklabels=np.around(y_std[::-1], decimals=2), annot=False)
        plt.xlabel(labels[1][0],size=20)
        plt.ylabel(labels[1][1], size=20)
        plt.title(labels[0][i], size=20)
    return figs

