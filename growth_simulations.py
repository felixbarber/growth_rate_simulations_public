import numpy as np
import weakref
import matplotlib.pyplot as plt
import seaborn as sns
import copy

par = dict([('num_s', 50), ('vd', 1.0), ('vm', 1.0), ('std_v', 0.1)])

# modeltype 0 is the model from cerulus et al., with an asymmetrically dividing, non-budding growth morphology defined
# by r
# modeltype 1 is a simple adder model, with an asymmetrically dividing, non-budding growth morphology defined by r
# modeltype 3 is the model from Amir 2014
list1 = [1, 4]  # 1 is asymmetric, 4 is asymmetric budding.
list2 = [4, 15]  # budding growth morphology
# modeltype 5 is an inhibitor synthesis model with a budding growth morphology
# modeltype 7 is a linear growth policy model f(vb) =2D+a*(vb-D) where a=0 gives sizer, a=1 gives adder and a=2 gives
# timer.
#modeltype 7.1 is a linear growth policy model f(vb) =2D+a*(vb-D) where a=0 gives sizer, a=1 gives adder and a=2 gives
# timer. The difference with 7 is that we enforce the growth policy strictly so that cells can decrease in volume
# modeltype 8 is an inhibitor dilution growth policy of the details matter form (inh produced = Delta), with a budding
# growth morphology, and where only a constant fraction f is kept after passage through Start


# modeltype 9 is an inhibitor dilution growth policy with a synthesis rate proportional to volume, such that kr=lambda
# This model has no noise in the length of the budded phase, and gives a fraction of inhibitor to the daughter cell that
# varies between r/(1+r) and all of it.

# modeltype 10 is an inhibitor dilution growth policy with a synthesis rate proportional to volume, and a rule that the
# first Delta of inhibitor goes to the daughter, with the remainder going to the mother

# testing for the size regulation meeting 5/21/19
# modeltype 11 is an inhibitor dilution growth policy with a constant synthesis rate k1 during S/G2/M phases. It also
# has the potential to have a non-size regulating component of the G1 phase
# parameters and units: lambda_std [], T2 [t], lambda [t-1], tg1_std [], r [], td_std [t], w_synth [V t-1], f[]

# modeltype 12 is an inhibitor dilution growth policy with a synthesis rate k1*V during S/G2/M phases. It also
# has the potential to have a non-size regulating component of the G1 phase
# parameters and units: lambda_std [], T2 [t], lambda [t-1], tg1_std [], r [], td_std [t], w_synth [V t-1]

# modeltype 13 is the same as modeltype 7 except that we allow for a variable growth rate of the form
# lambda~N(lambda_0*(1-(Vb/Delta-1)^n),sigma_lambda) where n can vary.

# modeltype 15 is the rate model for the inhibitor dilution paper, with a budding morphology
def fn(L, tm, td):
    return np.exp(-L * td) + np.exp(-L * tm) - 1.0

# models in which I track Whi5
global list_whi5
list_whi5 = [5, 8, 9, 10, 11, 12, 14, 15, 16]

def fn_1(L, tm, td):
    return (2*np.exp(L*tm)-3)/(2*(np.exp(L*tm)-1))-1/(2*(np.exp(L*td)-1))


class Cell(object):
    cellCount = 0  # total number of cells

    def __init__(self, birth_parameters, temp_params):  # birth parameters = [tb, celltype, parent, vb, parent_current]
        self.exists = True
        # these are present for all cells
        self.tb = birth_parameters[0]
        self.celltype = birth_parameters[1]  # 0 is mother, 1 is daughter
        self.parent = birth_parameters[2]
        self.vb = birth_parameters[3]
        self.parent_current = birth_parameters[4]  # only used for daughter cells
        self.neg_gr = False  # used to track which cells have a negative growth rate arising from being too large or too
        self.removed = False  # tracks whether the cell was removed through dilution
        # small in model 13
        if temp_params['modeltype'] in list_whi5:
            self.wb = birth_parameters[5]

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
        elif par1['modeltype'] in list1:  # simple adder model
            temp_zscore = np.random.normal(0.0, 1.0, size=1)[0]
            if par1['lambda_std'] is None:
                self.t_grow = max(np.log(1 + par1['delta'] / self.vb) / par1['lambda'] + temp_zscore*par1['td_std'][self.celltype],
                                  par1['dt'])
                # self.t_grow=np.log(1 + par1['delta'] / self.vb) / par1['lambda'] + temp_zscore * par1['td_std'][self.celltype]
                self.t_div = self.tb + self.t_grow
                self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            else:
                temp = np.random.normal(0.0, 1.0, size=1)[0]
                self.gr = par1['lambda']*(1+par1['lambda_std']*temp)
                self.t_grow = max(np.log(1 + par1['delta'] / self.vb) / self.gr + temp_zscore * par1['td_std'][
                    self.celltype], par1['dt'])
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
        elif par1['modeltype'] == 3:  # alpha tunable model from Amir 2014
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda']*(1+temp[0]*par1['lambda_std'])
            self.t_grow = max(np.log(2 * (par1['delta'] / self.vb)**(par1['alpha'])) / self.gr +
                              par1['td_std']*temp[1]/par1['lambda'], par1['dt'])
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        elif par1['modeltype'] == 5 or par1['modeltype']==6:  # inhibitor dilution with synth rate = a +bv
            temp = np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp)
            self.vs = np.amax([self.wb, self.vb])
            self.t_g2 = np.amax([0.0, np.log(1+par1['r'])/par1['lambda']+
                            np.random.normal(0.0,1.0,size=1)[0]*par1['td_std']/par1['lambda']])
            # timer for g2 where the average length is set by what we choose for the asymmetry ratio
            self.vd = self.vs*np.exp(self.gr*self.t_g2)
            self.wd = self.wb+par1['w_synth'][0]*self.t_g2+par1['w_synth'][1]*(self.vd-self.vs)/self.gr
            self.t_grow = np.log(self.vd/self.vb)/self.gr
            self.t_div = self.tb+self.t_grow
        elif par1['modeltype'] == 7:  # linear growth policy tuning between timer adder and sizer
            temp = np.random.normal(0.0, 1.0, size=2)
            while temp[0]<-1/par1['lambda_std']:  # ensuring a positive growth rate
                temp[0]=np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda']*(1+temp[0]*par1['lambda_std'])
            self.t_grow = max(np.log((2*par1['delta']+par1['alpha']*(self.vb-par1['delta']))/self.vb) / self.gr +
                              par1['td_std']*temp[1]/par1['lambda'], par1['dt'])
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        # self.t_div = self.tb + np.amax([self.t_grow, 0.0])
        elif par1['modeltype'] == 7.1:  # linear growth policy tuning between timer adder and sizer
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda']*(1+temp[0]*par1['lambda_std'])
            self.t_grow = np.log((2*par1['delta']+par1['alpha']*(self.vb-par1['delta']))/self.vb) / self.gr +\
                              par1['td_std']*temp[1]/par1['lambda']
            self.t_div = self.tb + max(self.t_grow,par1['dt'])
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        # self.t_div = self.tb + np.amax([self.t_grow, 0.0])
        elif par1['modeltype'] == 8:  # inhibitor dilution with perfectly asymmetric budding
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda'] * max(1 + temp[0] * par1['lambda_std'], 0.01)  # cannot have a negative growth rate
            self.t_g1 = max(np.log(self.wb/self.vb)/self.gr+par1['td_std']*temp[1]/par1['lambda'], par1['dt'])  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            self.t_g2 = max(np.log(1+par1['r'])/self.gr, par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = par1['f']*self.wb + par1['delta']  # no noise in inhibitor production, and only a fraction f
            # of inhibitor at birth is kept
        elif par1['modeltype']==9:
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda'] * max(1 + temp[0] * par1['lambda_std'], 0.01)  # cannot have a negative growth rate
            self.t_g1 = max(np.log(self.wb / self.vb) / self.gr + par1['td_std'] * temp[1] / par1['lambda'],
                            par1['dt'])  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            self.t_g2 = max(np.log(1 + par1['r']) / self.gr,
                            par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = self.wb + (self.vd-self.vs)/par1['r']
            # inhibitor is produced at a rate kV, with kr/lambda=1, so that dI=dV/r
        elif par1['modeltype']==10:
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda'] * max(1 + temp[0] * par1['lambda_std'], 0.01)  # cannot have a negative growth rate
            self.t_g1 = max(np.log(self.wb / self.vb) / self.gr + par1['td_std'] * temp[1] / par1['lambda'],
                            par1['dt'])  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            self.t_g2 = max(np.log(1 + par1['r']) / self.gr,
                            par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = self.wb + (self.vd - self.vs) / par1['r']
            # inhibitor is produced at a rate kV, with kr/lambda=1, so that dI=dV/r

        elif par1['modeltype']==11: # inhibitor dilution with synth rate = k1
            # parameters and units: lambda_std [], T2 [t], lambda [t-1], tg1_std [], r [], td_std [t], w_synth [V t-1]
            #f[]
            temp = np.random.normal(0.0, 1.0, size=3)
            self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp[0])
            self.vs = np.amax([self.wb, self.vb])*np.exp(self.gr*par1['T2']*(1+par1['tg1_std']*temp[1]))
            self.vd = self.vs*(1+par1['r'])*np.exp(self.gr*temp[2]*par1['td_std'])
            # timer for g2 where the average length is set by what we choose for the asymmetry ratio
            self.t_g2 = np.amax([par1['dt'], np.log(self.vd/self.vs)/self.gr])
            self.wd = self.wb + par1['w_synth'] * self.t_g2
            self.t_grow = np.amax([par1['dt'], np.log(self.vd / self.vb) / self.gr])
            self.t_div = self.tb + self.t_grow
        elif par1['modeltype']==12: # inhibitor dilution with synth rate = k1*V
            # parameters and units: lambda_std [], T2 [t], lambda [t-1], tg1_std [], r [], td_std [t], w_synth [V t-1]
            #f[]
            temp = np.random.normal(0.0, 1.0, size=3)
            self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp[0])
            self.vs = np.amax([self.wb, self.vb])*np.exp(self.gr*par1['T2']*(1+par1['tg1_std']*temp[1]))
            self.vd = self.vs*(1+par1['r'])*np.exp(self.gr*temp[2]*par1['td_std'])
            # timer for g2 where the average length is set by what we choose for the asymmetry ratio
            self.t_g2 = np.amax([par1['dt'], np.log(self.vd/self.vs)/self.gr])
            self.wd = self.wb + par1['w_synth'] * (self.vd-self.vs)/self.gr
            self.t_grow = np.amax([par1['dt'], np.log(self.vd / self.vb) / self.gr])
            self.t_div = self.tb + self.t_grow
        elif par1['modeltype'] == 13:  # linear growth policy tuning between timer adder and sizer with variable growth
            # rate depending on size at birth.
            temp = np.random.normal(0.0, 1.0, size=2)
            temp_gr_mean = 1 - par1['epsilon']*(self.vb / par1['delta'] - 1) ** par1['exp_n']
            while temp[0] < -temp_gr_mean / par1['lambda_std']:  # ensuring a positive growth rate
                temp[0] = np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda'] * (temp_gr_mean + temp[0] * par1['lambda_std'])
            self.t_grow = max(
                np.log((2 * par1['delta']*par1['alpha']+2*(1-par1['alpha'])*self.vb ) / self.vb) / self.gr +
                par1['td_std'] * temp[1] / par1['lambda'], par1['dt'])
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        elif par1['modeltype'] == 14:  # inhibitor dilution with perfectly asymmetric budding
            temp = np.random.normal(0.0, 1.0, size=2)
            temp_av_v = (1+par1['r'])*par1['delta']/(2*(2-par1['f']))
            temp_gr_mean = 1 - par1['epsilon'] * (self.vb / temp_av_v - 1) ** par1['exp_n']
            while temp[0] < -temp_gr_mean / par1['lambda_std']:  # ensuring a positive growth rate
                temp[0] = np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda'] * (temp_gr_mean + temp[0] * par1['lambda_std'])
            self.t_g1 = max(np.log(self.wb/self.vb)/self.gr+par1['td_std']*temp[1]/par1['lambda'], par1['dt'])  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            # self.t_g2 = max(np.log(1 + par1['r']) / par1['lambda'],
            #                 par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            self.t_g2 = max(np.log(1 + par1['r']) / self.gr,
                            par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = par1['f']*self.wb + par1['delta']  # no noise in inhibitor production, and only a fraction f
            # of inhibitor at birth is kept
        elif par1['modeltype'] == 15:  # rate model with budding growth morphology
            # parameters = k, n
            # noisy rate model for passage through Start with no whi5 synthesis during the G1 phase
            self.gr = par1['lambda']
            temp_grow = 0.0
            temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            while temp > par1['k'] * self.vb ** par1['n'] * np.exp(par1['n'] * par1['lambda'] * temp_grow) * 0.01 / self.wb ** par1['n']:
                temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                temp_grow += par1['dt']
            self.t_g1 = temp_grow+par1['dt']  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(par1['lambda']* self.t_g1)  # sizer at passage through Start
            self.t_g2 = np.log(1 + par1['r']) / par1['lambda']
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            self.wd = self.wb + par1['delta']  # no noise in inhibitor production
        elif par1['modeltype'] == 16:  # rate model with budding growth morphology
            # parameters = kw, k, n
            # noisy rate model for passage through Start with no whi5 synthesis during the G1 phase
            self.gr = par1['lambda']
            temp_grow = 0.0
            temp_w = self.wb
            temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            while temp > par1['k'] * self.vb ** par1['n'] * np.exp(par1['n'] * par1['lambda'] * temp_grow) * 0.01 / temp_w ** par1['n']:
                temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                temp_grow += par1['dt']
                temp_w+=par1['kw']*self.vb*np.exp(par1['lambda'] * temp_grow)*par1['dt']  # additional Whi5 synthesized
            self.t_g1 = temp_grow +par1['dt'] # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(par1['lambda']* self.t_g1)  # sizer at passage through Start
            self.t_g2 = np.log(1 + par1['r']) / par1['lambda']
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            self.wd = self.wb + par1['kw']*(self.vd-self.vb)/par1['lambda']  # no noise in inhibitor production
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
        elif par1['modeltype'] in list1:  # simple adder model
            temp_zscore = np.random.normal(0.0, 1.0, size=1)[0]
            if par1['lambda_std'] is None:
                self.t_grow = max(np.log(1.0 + par1['delta'] / self.vb) / par1['lambda'] + temp_zscore * par1['td_std'][
                    self.celltype], par1['dt'])
                # self.t_grow=np.log(1 + par1['delta'] / self.vb) / par1['lambda'] + temp_zscore * par1['td_std'][self.celltype]
                self.t_div = self.tb + self.t_grow
                self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            else:
                temp = np.random.normal(0.0, 1.0, size=1)[0]
                self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp)
                self.t_grow = max(np.log(1 + par1['delta'] / self.vb) / self.gr + temp_zscore * par1['td_std'][
                    self.celltype], par1['dt'])
                # self.t_grow = np.log(1 + par1['delta'] / self.vb) / self.gr + temp_zscore * par1['td_std'][
                #     self.celltype]
                self.t_div = self.tb + self.t_grow
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
        elif par1['modeltype'] == 3:  # alpha tunable model from Amir 2014
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda']*(1+temp[0]*par1['lambda_std'])
            self.t_grow = max(np.log(2 * (par1['delta'] / self.vb)**(par1['alpha'])) / self.gr +
                              par1['td_std']*temp[1]/par1['lambda'], par1['dt'])
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        elif par1['modeltype'] == 5 or par1['modeltype'] == 6:  # inhibitor dilution with synth rate = a+b*v
            temp = np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp)
            self.vs = np.amax([self.wb, self.vb])
            self.t_g2 = np.amax([0.0, np.log(1+par1['r'])/par1['lambda']+
                            np.random.normal(0.0,1.0,size=1)[0]*par1['td_std']/par1['lambda']])
            # timer for g2 where the average length is set by what we choose for the asymmetry ratio
            self.vd = self.vs*np.exp(self.gr*self.t_g2)
            self.t_grow = np.log(self.vd / self.vb) / self.gr
            self.wd = self.wb+par1['w_synth'][0]*self.t_g2+par1['w_synth'][1]*(self.vd-self.vs)/self.gr
            self.t_div = self.tb+self.t_grow
        elif par1['modeltype'] == 7:  # inhibitor dilution with synth rate = a +bv
            temp = np.random.normal(0.0, 1.0, size=2)
            while temp[0]<-1/par1['lambda_std']:  # ensuring a positive growth rate
                temp[0]=np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda']*(1+temp[0]*par1['lambda_std'])
            self.t_grow = max(np.log((2*par1['delta']+par1['alpha']*(self.vb-par1['delta']))/self.vb) / self.gr +
                              par1['td_std']*temp[1]/par1['lambda'], par1['dt'])
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        elif par1['modeltype'] == 7.1:  # linear growth policy tuning between timer adder and sizer
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda']*(1+temp[0]*par1['lambda_std'])
            self.t_grow = np.log((2*par1['delta']+par1['alpha']*(self.vb-par1['delta']))/self.vb) / self.gr +\
                              par1['td_std']*temp[1]/par1['lambda']
            self.t_div = self.tb + max(self.t_grow,par1['dt'])
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        elif par1['modeltype'] == 8:  # inhibitor dilution with perfectly asymmetric budding
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda'] * max(1 + temp[0] * par1['lambda_std'], 0.01)  # cannot have a negative growth rate
            self.t_g1 = max(np.log(self.wb / self.vb) / self.gr + par1['td_std'] * temp[1] / par1['lambda'], par1['dt'])
            # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            self.t_g2 = max(np.log(1 + par1['r']) / self.gr,
                            par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = par1['f'] * self.wb + par1['delta']  # no noise in inhibitor production, and only a fraction f
            # of inhibitor at birth is kept
        elif par1['modeltype']==9:
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda'] * max(1 + temp[0] * par1['lambda_std'], 0.01)  # cannot have a negative growth rate
            self.t_g1 = max(np.log(self.wb / self.vb) / self.gr + par1['td_std'] * temp[1] / par1['lambda'],
                            par1['dt'])  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            self.t_g2 = max(np.log(1 + par1['r']) / self.gr,
                            par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = self.wb + (self.vd-self.vs)/par1['r']
            # inhibitor is produced at a rate kV, with kr/lambda=1, so that dI=dV/r
        elif par1['modeltype']==10:
            temp = np.random.normal(0.0, 1.0, size=2)
            self.gr = par1['lambda'] * max(1 + temp[0] * par1['lambda_std'], 0.01)  # cannot have a negative growth rate
            self.t_g1 = max(np.log(self.wb / self.vb) / self.gr + par1['td_std'] * temp[1] / par1['lambda'],
                            par1['dt'])  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            self.t_g2 = max(np.log(1 + par1['r']) / self.gr,
                            par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = self.wb + (self.vd - self.vs) / par1['r']
            # inhibitor is produced at a rate kV, with kr/lambda=1, so that dI=dV/r
        elif par1['modeltype']==11: # inhibitor dilution with synth rate = k1
            # parameters and units: lambda_std [], T2 [t], lambda [t-1], tg1_std [], r [], td_std [t], w_synth [V t-1]
            #f[]
            temp = np.random.normal(0.0, 1.0, size=3)
            self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp[0])
            self.vs = np.amax([self.wb, self.vb])*np.exp(self.gr*par1['T2']*(1+par1['tg1_std']*temp[1]))
            self.vd = self.vs*(1+par1['r'])*np.exp(self.gr*temp[2]*par1['td_std'])
            # timer for g2 where the average length is set by what we choose for the asymmetry ratio
            self.t_g2 = np.amax([par1['dt'], np.log(self.vd/self.vs)/self.gr])
            self.wd = self.wb + par1['w_synth'] * self.t_g2
            self.t_grow = np.amax([par1['dt'], np.log(self.vd / self.vb) / self.gr])
            self.t_div = self.tb + self.t_grow
        elif par1['modeltype']==12: # inhibitor dilution with synth rate = k1*V
            # parameters and units: lambda_std [], T2 [t], lambda [t-1], tg1_std [], r [], td_std [t], w_synth [V t-1]
            #f[]
            temp = np.random.normal(0.0, 1.0, size=3)
            self.gr = par1['lambda'] * (1 + par1['lambda_std'] * temp[0])
            self.vs = np.amax([self.wb, self.vb])*np.exp(self.gr*par1['T2']*(1+par1['tg1_std']*temp[1]))
            self.vd = self.vs*(1+par1['r'])*np.exp(self.gr*temp[2]*par1['td_std'])
            # timer for g2 where the average length is set by what we choose for the asymmetry ratio
            self.t_g2 = np.amax([par1['dt'], np.log(self.vd/self.vs)/self.gr])
            self.wd = self.wb + par1['w_synth'] * (self.vd-self.vs)/self.gr
            self.t_grow = np.amax([par1['dt'], np.log(self.vd / self.vb) / self.gr])
            self.t_div = self.tb + self.t_grow
        elif par1['modeltype'] == 13:  # linear growth policy tuning between timer adder and sizer with variable growth
            # rate depending on size at birth.
            temp = np.random.normal(0.0, 1.0, size=2)
            temp_gr_mean = np.amax([1 - par1['epsilon']*(self.vb / par1['delta'] - 1) ** par1['exp_n'],0.0])
            if temp_gr_mean==0:
                self.neg_gr=True  # this way we keep track of whether this cell had a negative growth rate
            while temp[0]<-temp_gr_mean/par1['lambda_std']:  # ensuring a positive growth rate
                temp[0]=np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda']*(temp_gr_mean+temp[0]*par1['lambda_std'])

            self.t_grow = max(np.log((2 * par1['delta']*par1['alpha']+2*(1-par1['alpha'])*self.vb ) / self.vb) / self.gr +
                              par1['td_std']*temp[1]/par1['lambda'], par1['dt'])
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
        elif par1['modeltype'] == 14:  # inhibitor dilution with perfectly asymmetric budding
            temp = np.random.normal(0.0, 1.0, size=2)
            temp_av_v = (1 + par1['r']) * par1['delta'] / (2 * (2 - par1['f']))
            temp_gr_mean = 1 - par1['epsilon'] * (self.vb / temp_av_v - 1) ** par1['exp_n']
            while temp[0] < -temp_gr_mean / par1['lambda_std']:  # ensuring a positive growth rate
                temp[0] = np.random.normal(0.0, 1.0, size=1)[0]
            self.gr = par1['lambda'] * (temp_gr_mean + temp[0] * par1['lambda_std'])
            self.t_g1 = max(np.log(self.wb/self.vb)/self.gr+par1['td_std']*temp[1]/par1['lambda'], par1['dt'])  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(self.gr * self.t_g1)  # sizer at passage through Start
            # self.t_g2 = max(np.log(1 + par1['r']) / par1['lambda'],
            #                 par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            self.t_g2 = max(np.log(1+par1['r'])/self.gr, par1['dt'])  # assumes a constant asymmetry between mother and daughter, so that
            # it is consistent with other non-budding models which have no division asymmetry noise.
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(self.gr * self.t_grow)
            self.wd = par1['f']*self.wb + par1['delta']  # no noise in inhibitor production, and only a fraction f
            # of inhibitor at birth is kept
        elif par1['modeltype'] == 15:  # rate model with budding growth morphology
            # parameters = k, n
            # noisy rate model for passage through Start with no whi5 synthesis during the G1 phase
            self.gr = par1['lambda']
            temp_grow = 0.0
            temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            while temp > par1['k'] * self.vb ** par1['n'] * np.exp(par1['n'] * par1['lambda'] * temp_grow) * 0.01 / self.wb ** par1['n']:
                temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                temp_grow += par1['dt']
            self.t_g1 = temp_grow+par1['dt']  # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(par1['lambda']* self.t_g1)  # sizer at passage through Start
            self.t_g2 = np.log(1 + par1['r']) / par1['lambda']
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            self.wd = self.wb + par1['delta']  # no noise in inhibitor production
        elif par1['modeltype'] == 16:  # rate model with budding growth morphology
            #parameters = kw, k, n
            # noisy rate model for passage through Start with no whi5 synthesis during the G1 phase
            self.gr = par1['lambda']
            temp_grow = 0.0
            temp_w = self.wb
            temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            while temp > par1['k'] * self.vb ** par1['n'] * np.exp(par1['n'] * par1['lambda'] * temp_grow) * 0.01 / temp_w ** par1['n']:
                temp = np.random.uniform(low=0.0, high=1.0, size=1)[0]
                temp_grow += par1['dt']
                temp_w+=par1['kw']*self.vb*np.exp(par1['lambda'] * temp_grow)*par1['dt']  # additional Whi5 synthesized
            self.t_g1 = temp_grow +par1['dt'] # as a minimum this occurs one timestep later
            self.vs = self.vb * np.exp(par1['lambda']* self.t_g1)  # sizer at passage through Start
            self.t_g2 = np.log(1 + par1['r']) / par1['lambda']
            self.t_grow = self.t_g2 + self.t_g1
            self.t_div = self.tb + self.t_grow
            self.vd = self.vb * np.exp(par1['lambda'] * self.t_grow)
            self.wd = self.wb + par1['kw']*(self.vd-self.vb)/par1['lambda']  # no noise in inhibitor production
        Cell.cellCount += 1

    def size(self, par1, t):  # this evaluates the volume of this cell at a particular point in time
        # if par1['modeltype']==1:
        if par1['lambda_std'] is None:
            temp = par1['lambda']
        else:
            temp = self.gr
        temp1 = self.vb*np.exp(temp*(t-self.tb))
        return temp1
    def is_in_g1(self,par1, t):  # this evaluates roughly whether the cell is in G1 or not
        if par1['lambda_std'] is None:
            temp = par1['lambda']
        else:
            temp = self.gr
        temp1 = t-self.tb<self.t_grow-np.log(1+par1['r'])/temp  # this returns a true value if at the current timepoint
        # the cell should be in G1
        return temp1


def starting_popn(par1):
    # print 'Hi'
    # To clarify we first set the initial condition for the simulation.
    l = par1['lambda']
    if par1['modeltype'] == 0:  # in this case, there is no size regulation so av cell size is set by the initial
        # condition. Note that by contrast with model 1, we have the same relative scaling between vd and vm to begin.
        vd = par1['r']*par1['v_init']
        vm = par1['v_init']
    elif par1['modeltype'] in list1:  # set the average size distributions to begin with here.
        vm = 2.0*par1['delta']/(1+par1['r'])
        vd = 2*par1['r']*par1['delta']/(1+par1['r'])
    elif par1['modeltype'] == 2:  # set the average size distributions to begin with here.
        vm = par1['delta']/(1+par1['r'])
        vd = par1['r']*par1['delta']/(1+par1['r'])
    elif par1['modeltype'] == 3:  # set the average size distributions to begin with here.
        vm = par1['delta']/(1+par1['r'])
        vd = par1['r']*par1['delta']/(1+par1['r'])
    elif par1['modeltype'] == 5:
        temp = par1['w_synth'][0]*np.log(1+par1['r'])/(par1['lambda']-par1['w_synth'][1]*par1['r'])
        vm = temp
        vd = par1['r']*temp
    elif par1['modeltype'] == 6:
        vm = par1['init_v']
        vd = par1['r']*par1['init_v']
    elif par1['modeltype'] == 7 or par1['modeltype']==13:
        vm = par1['delta'] / (1 + par1['r'])
        vd = par1['r'] * par1['delta'] / (1 + par1['r'])
    elif par1['modeltype'] == 7.1:
        vm = par1['delta'] / (1 + par1['r'])
        vd = par1['r'] * par1['delta'] / (1 + par1['r'])
    elif par1['modeltype'] == 8 or par1['modeltype'] == 9 or par1['modeltype'] == 10 or par1['modeltype']==14:
        temp = par1['delta']/(2-par1['f'])
        vm = temp*np.exp(par1['T2']*par1['lambda'])
        vd = par1['r']*temp*np.exp(par1['T2']*par1['lambda'])
    elif par1['modeltype'] == 11 or par1['modeltype'] == 12:
        temp = par1['w_synth']*np.log(1+par1['r'])/par1['lambda']
        vm = temp
        vd =par1['r']*temp
    elif par1['modeltype'] in [15,16]:  # can't solve these analytically so just seed them with something which looks
        # fine based on the average volumes
        vm = par1['delta']*1.5 / (1 + par1['r'])
        vd = par1['delta']*1.5*par1['r'] / (1 + par1['r'])
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
        # print 'Hi'
        temp = [0.0, 1, None, v_init_d[i], None]
        if par1['modeltype']==5:
            temp.append(par1['r']*par1['w_synth'][0]*np.log(1+par1['r'])/((par1['lambda']-
                                                                           par1['w_synth'][1]*par1['r'])*(1+par1['r'])))
        elif par1['modeltype']==6:  # in this case there is zero base synthesis rate of Whi5
            temp.append(par1['init_w']/(1.0+par1['r']))
        elif par1['modeltype'] == 8 or par1['modeltype']==14:  # inhibitor amount
            temp.append(2 * par1['delta']*par1['r']/((2-par1['f']) *(1 + par1['r'])))
        elif par1['modeltype'] == 9:  # inhibitor amount with constant frac
            temp.append(2*par1['delta'] * par1['f'])
        elif par1['modeltype'] == 10:  # inhibitor amount with constant amt given to daughter
            temp.append(par1['delta'])
        elif par1['modeltype'] == 11 or par1['modeltype'] == 12:
            temp.append(2*par1['f']*par1['w_synth']*np.log(1+par1['r'])/par1['lambda'])
        elif par1['modeltype'] in [15,16]:
            temp.append(2*par1['delta']*par1['r']/(1+par1['r']))
        # print temp
        c.append(Cell(temp, par1))
        c[-1].grow_g1(par1)
        c[-1].t_div *= (1.0-t_div[i])
        c[-1].tb = c[-1].t_div - c[-1].t_grow
        c[-1].exists=True
        # Mothers
        temp = [0.0, 0, None, v_init_m[i], None]
        if par1['modeltype']==5:
            temp.append(par1['w_synth'][0]*np.log(1+par1['r'])/((par1['lambda']-
                                                                           par1['w_synth'][1]*par1['r'])*(1+par1['r'])))
        elif par1['modeltype']==6:  # in this case there is zero base synthesis rate of Whi5
            temp.append(par1['init_w']*par1['r']/(1.0+par1['r']))
        elif par1['modeltype'] == 8 or par1['modeltype']==14:  # inhibitor amount
            temp.append(2*par1['delta']/((2-par1['f'])*(1+par1['r'])))
        elif par1['modeltype'] == 9:  # inhibitor amount with constant frac
            temp.append(2 * par1['delta'] * (1-par1['f']))
        elif par1['modeltype'] == 10:  # inhibitor amount with constant amt given to daughter
            temp.append(par1['delta'])
        elif par1['modeltype'] == 11 or par1['modeltype'] == 12:
            temp.append(2*(1-par1['f'])*par1['w_synth']*np.log(1+par1['r'])/par1['lambda'])
        elif par1['modeltype'] in [15,16]:
            temp.append(2 * par1['delta']/ (1 + par1['r']))
        c.append(Cell(temp, par1))
        c[-1].grow_g1(par1)
        c[-1].t_div *= (1.0 - t_div[i])
        c[-1].tb = c[-1].t_div - c[-1].t_grow
        c[-1].exists = True
        # defined in this manner all starting cells have been born at time less than or equal to 0.
    del v_init_d, v_init_m
    return c


    # how to set growth rate in such a way that the size distribution doesn't diverge? Implies that growth rate is tuned?
    # Seems unrealistic. May be instead that growth rate is modulated.


def next_gen(index, f, t, par1,temp_ind=-1):
    # This function resets growth-policy specific variables for a single birth event.
    # Should be used within discr_time to evolve the list of cells c.
    # frac = max((f[index].vd-f[index].vi)/f[index].vd, 0.0)

    if par1['modeltype'] in list2:
        frac1 = min(par1['r']/(1+par1['r']), (f[index].vd-f[index].vb)/f[index].vd)  # ensures that cells do not
        # shrink over subsequent generations (i.e. that newborn daughter size is at most the new growth added)
    elif par1['modeltype']==5:
        frac1 = (f[index].vd-f[index].vb)/f[index].vd
    elif par1['modeltype'] == 8 or par1['modeltype']==14:
        frac1 = par1['r']/(1+par1['r'])  # should be r/(1+r)
    elif par1['modeltype'] == 9:
        frac1 = par1['r']/(1+par1['r'])
        frac2 = par1['f']
    elif par1['modeltype'] == 11 or par1['modeltype'] == 12:
        frac1 = (f[index].vd-f[index].vs)/f[index].vd
        frac2 = par1['f']
    else:
        frac1 = par1['r']/(1+par1['r'])

    # add new cell for new cycle of mother cell.
    temp = [t, 0, weakref.proxy(f[index]), (1-frac1) * f[index].vd, None]
    if par1['modeltype'] == 5 or par1['modeltype'] == 6 or par1['modeltype'] == 8 or par1['modeltype']==14 or par1['modeltype'] in [15,16]:
        temp.append((1-frac1)*f[index].wd)
    elif par1['modeltype'] == 9:
        temp.append((1-frac2)*f[index].wd)
    elif par1['modeltype'] == 10:
        temp.append(np.amax([f[index].wd-par1['Id']*par1['delta'],0.01]))
    elif par1['modeltype'] == 11 or par1['modeltype'] == 12:
        temp.append((1 - frac2) * f[index].wd)
        # temp.append((1-par1['frac_w']) * f[index].wd)
    # if temp_ind > 250:
    #     print 'I got to part A1 of step number {0}'.format(temp_ind)
    f.append(Cell(temp, par1))
    # if temp_ind > 250:
    #     print 'I got to part A2 of step number {0}'.format(temp_ind)
    f[-1].grow(par1)  # grow newborn cell
    # if temp_ind > 250:
    #     print 'I got to part A3 of step number {0}'.format(temp_ind)
    f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
    f[index].exists = False  # track that this cell no longer "exists".
    # daughters
    temp = [t, 1, weakref.proxy(f[index]), frac1*f[index].vd, weakref.proxy(f[-1])]
    if par1['modeltype'] == 5 or par1['modeltype'] == 6 or par1['modeltype'] == 8 or par1['modeltype'] == 14 or par1['modeltype'] in [15,16]:
        temp.append(frac1*f[index].wd)
    elif par1['modeltype'] == 9:
        temp.append(frac2* f[index].wd)
    elif par1['modeltype'] == 10:
        temp.append(np.amin([f[index].wd, par1['Id'] * par1['delta']]))
    elif par1['modeltype'] == 11 or par1['modeltype'] == 12:
        temp.append(frac2* f[index].wd)
        # temp.append(par1['frac_w'] * f[index].wd)
    f.append(Cell(temp, par1))
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
    f.append(Cell(temp, par1))
    f[-1].grow(par1)  # grow newborn cell
    f[index].nextgen = weakref.proxy(f[-1])  # track that this cell is the next generation of the the current cell.
    f[index].exists = False  # track that this cell no longer "exists".
    # daughters
    temp = [t, 1, weakref.proxy(f[index]), frac1*f[index].vd, weakref.proxy(f[-1])]
    temp1 = Cell(temp, par1)
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
    num_existent_mothers = np.zeros(tvec.shape)
    num_existent_daughters = np.zeros(tvec.shape)
    vol_existent_mothers = np.zeros(tvec.shape)
    vol_existent_daughters = np.zeros(tvec.shape)
    num_div_d = np.zeros(tvec.shape)  # keep track of the number of divisions from mother cells
    num_div_m = np.zeros(tvec.shape)  # keep track of the number of divisions from daughter cells
    av_v = np.zeros(tvec.shape)
    std_v = np.zeros(tvec.shape)
    av_log_v = np.zeros(tvec.shape)
    std_log_v = np.zeros(tvec.shape)
    std_v_pop = [np.zeros(tvec.shape),np.zeros(tvec.shape)]
    av_v_pop = [np.zeros(tvec.shape), np.zeros(tvec.shape)]
    av_v_async = np.zeros(tvec.shape)
    std_v_async = np.zeros(tvec.shape)
    av_v_async_G1 = np.zeros(tvec.shape)
    std_v_async_G1 = np.zeros(tvec.shape)
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
        # if np.mod(i,10)==0 and i>400:
        #     print 'I got to step number {0}, with {1} cells'.format(i, len(c))
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
            # if i > 250:
            #     print 'I got to step number {0}, with {1} cells'.format(i, len(c))
            c = next_gen(index, c, c[index].t_div, par1)  # compute the next generation for this cell.
            # if i > 250:
            #     print 'I got to part B of step number {0}, with {1} cells'.format(i, len(c))
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
        temp_vols = [obj.size(par1, tvec[i]) for obj in c if obj.exists]
        temp_vols_G1 = [obj.size(par1, tvec[i]) for obj in c if obj.exists and obj.is_in_g1(par1,tvec[i])]
        num_cells[i] = len(c)
        temp_val = [obj.vb for obj in c if obj.exists]
        av_v[i] = np.mean(temp_val)
        std_v[i] = np.std(temp_val)
        av_log_v[i] = np.mean(np.log(temp_val))
        std_log_v[i] = np.std(np.log(temp_val))
        temp_val1 = [obj.vb for obj in c if obj.exists and obj.celltype == 0]
        temp_val2 = [obj.vb for obj in c if obj.exists and obj.celltype == 1]
        std_v_pop[0][i] = np.std(temp_val1)
        std_v_pop[1][i] = np.std(temp_val2)
        av_v_pop[0][i] = np.mean(temp_val1)
        av_v_pop[1][i] = np.mean(temp_val2)
        num_existent_cells[i] = len(temp_val)
        num_existent_mothers[i] = len(temp_val1)
        num_existent_daughters[i] = len(temp_val2)
        vol_existent_mothers[i] = np.sum([obj.size(par1, tvec[i]) for obj in c if obj.exists and obj.celltype==0])
        vol_existent_daughters[i] = np.sum([obj.size(par1, tvec[i]) for obj in c if obj.exists and obj.celltype == 1])
        av_v_async[i] = np.mean(temp_vols)
        std_v_async[i] = np.std(temp_vols)
        av_v_async_G1[i] = np.mean(temp_vols_G1)
        std_v_async_G1[i] = np.std(temp_vols_G1)
        num_neg_gr_full = 1.0*np.sum([obj.neg_gr for obj in c if obj.exists])/len([obj for obj in c if obj.exists])
        num_neg_gr_mothers = 1.0 * np.sum([obj.neg_gr for obj in c if obj.exists and obj.celltype==0]) / len([obj for obj in c if obj.exists and obj.celltype==0])
        num_neg_gr_daughters = 1.0 * np.sum([obj.neg_gr for obj in c if obj.exists and obj.celltype == 1]) / len(
            [obj for obj in c if obj.exists and obj.celltype == 1])
    obs = [num_cells, tvec, num_div_d, num_div_m, num_existent_cells, av_v, std_v, vol, std_v_pop, av_v_pop,
           num_existent_mothers, num_existent_daughters, vol_existent_mothers, vol_existent_daughters, av_v_async,
           std_v_async, av_v_async_G1, std_v_async_G1, [num_neg_gr_mothers, num_neg_gr_daughters,num_neg_gr_full],
           av_log_v, std_log_v]
    return c, obs, [temp_vols,temp_vols_G1]


def discr_time_dilution(par1, starting_pop):
    # This function will simulate a full population of cells growing in a discretized time format and give us all the
    # info we need about the final population. Inputs are a set of parameters par1 and a starting population of cells.
    # This differs from discr_time_1 by also simulating the dilution of cells once the population number gets to be
    # above a certain threshold, giving a more smooth simulation of the population growth
    nstep = par1['nstep']
    tvec = np.linspace(0.0, nstep * par1['dt'] * np.log(2)/par1['lambda'], nstep + 1)
    num_cells = np.zeros(tvec.shape)
    num_existent_cells = np.zeros(tvec.shape)
    num_existent_mothers = np.zeros(tvec.shape)
    num_existent_daughters = np.zeros(tvec.shape)
    vol_existent_mothers = np.zeros(tvec.shape)
    vol_existent_daughters = np.zeros(tvec.shape)
    num_div_d = np.zeros(tvec.shape)  # keep track of the number of divisions from mother cells
    num_div_m = np.zeros(tvec.shape)  # keep track of the number of divisions from daughter cells
    av_v = np.zeros(tvec.shape)
    std_v = np.zeros(tvec.shape)
    av_log_v = np.zeros(tvec.shape)
    std_log_v = np.zeros(tvec.shape)
    std_v_pop = [np.zeros(tvec.shape),np.zeros(tvec.shape)]
    av_v_pop = [np.zeros(tvec.shape), np.zeros(tvec.shape)]
    av_v_async = np.zeros(tvec.shape)
    std_v_async = np.zeros(tvec.shape)
    av_v_async_G1 = np.zeros(tvec.shape)
    std_v_async_G1 = np.zeros(tvec.shape)
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
    cell_num_tracker = len(c)
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
        # if np.mod(i,10)==0 and i>400:
        #     print 'I got to step number {0}, with {1} cells'.format(i, len(c))
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
            # if i > 250:
            #     print 'I got to step number {0}, with {1} cells'.format(i, len(c))
            if c[index].exists:  # since we now randomly remove cells, it may be that this cell was removed before it
                # divided, in which case it will be marked as not existing anymore and we must not grow it to avoid
                # making the population too big
                c = next_gen(index, c, c[index].t_div, par1)  # compute the next generation for this cell.
                cell_num_tracker+=1
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

        if cell_num_tracker > par1['Nmax']:  # whenever the population size gets too big we
            # remove cells
            temp_cells = np.nonzero([obj.exists for obj in c])[0] # gives us the currently existing cells
            to_be_removed = []
            while len(to_be_removed)<cell_num_tracker-par1['Nmax']:
                temp_index = np.random.randint(low=0,high=len(temp_cells),size=1)[0]
                if not(temp_index in to_be_removed):
                    to_be_removed.append(temp_index)
            for ind in to_be_removed:
                c[temp_cells[ind]].exists=False
                c[temp_cells[ind]].removed = True
                cell_num_tracker += -1
        if np.mod(i,100)==0:
            print 'Reached step n={0}, population size {1}, total number of tracked cells {2}, number of diluted cells {3}'.\
                format(i, len([1 for obj in c if obj.exists]), len(c),len([1 for obj in c if obj.removed]))

        temp_cells = np.nonzero([obj.exists for obj in c])[0]  # gives us the currently existing cells

        vol[i] = np.sum([c[ind].size(par1, tvec[i]) for ind in temp_cells])  # cumulative size of the cells at each
        # timestep
        temp_vols = [c[ind].size(par1, tvec[i]) for ind in temp_cells]
        temp_vols_G1 = [c[ind].size(par1, tvec[i]) for ind in temp_cells if c[ind].is_in_g1(par1,tvec[i])]
        num_cells[i] = len(c)
        temp_val = [c[ind].vb for ind in temp_cells]
        av_v[i] = np.mean(temp_val)
        std_v[i] = np.std(temp_val)
        av_log_v[i] = np.mean(np.log(temp_val))
        std_log_v[i] = np.std(np.log(temp_val))
        temp_val1 = [c[ind].vb for ind in temp_cells if c[ind].celltype == 0]
        temp_val2 = [c[ind].vb for ind in temp_cells if c[ind].celltype == 1]
        std_v_pop[0][i] = np.std(temp_val1)
        std_v_pop[1][i] = np.std(temp_val2)
        av_v_pop[0][i] = np.mean(temp_val1)
        av_v_pop[1][i] = np.mean(temp_val2)
        num_existent_cells[i] = len(temp_val)
        num_existent_mothers[i] = len(temp_val1)
        num_existent_daughters[i] = len(temp_val2)
        vol_existent_mothers[i] = np.sum([c[ind].size(par1, tvec[i]) for ind in temp_cells if c[ind].celltype==0])
        vol_existent_daughters[i] = np.sum([c[ind].size(par1, tvec[i]) for ind in temp_cells if c[ind].celltype == 1])
        av_v_async[i] = np.mean(temp_vols)
        std_v_async[i] = np.std(temp_vols)
        av_v_async_G1[i] = np.mean(temp_vols_G1)
        std_v_async_G1[i] = np.std(temp_vols_G1)
        num_neg_gr_full = 1.0*np.sum([c[ind].neg_gr for ind in temp_cells])/len(temp_cells)
        num_neg_gr_mothers = 1.0 * np.sum([c[ind].neg_gr for ind in temp_cells if c[ind].celltype==0]) / len([1 for ind in temp_cells if c[ind].celltype==0])
        num_neg_gr_daughters = 1.0 * np.sum([c[ind].neg_gr for ind in temp_cells if c[ind].celltype==1]) / len(
            [1 for ind in temp_cells if c[ind].celltype == 1])
    obs = [num_cells, tvec, num_div_d, num_div_m, num_existent_cells, av_v, std_v, vol, std_v_pop, av_v_pop,
           num_existent_mothers, num_existent_daughters, vol_existent_mothers, vol_existent_daughters, av_v_async,
           std_v_async, av_v_async_G1, std_v_async_G1, [num_neg_gr_mothers, num_neg_gr_daughters,num_neg_gr_full],
           av_log_v, std_log_v]
    return c, obs, [temp_vols,temp_vols_G1]


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


def starting_popn_seeded(temp_cells, temp_params, discr_time=True):
    temp_cells = [obj for obj in temp_cells if obj.exists]
    indices = np.random.randint(low=0, high=len(temp_cells), size=8*par['num_s'])
    temp = [temp_cells[ind] for ind in indices]
    val = []

    for obj in temp:  # copy cells this way to avoid changing the properties of the population seeded from.
        val.append(copy.copy(obj))
        # temp1 = [obj.tb, obj.celltype, obj.parent, obj.vb, obj.parent_current, obj.wb]
        # val.append(Cell(birth_parameters=temp1, temp_params=temp_params))  # note that depending on how this code is implemented,
        # # obj.mother may not exist. We therefore do the following.
        # # reset all the variables to be the same as that in the previous cell, so we don't have to 'grow' them again.
        # val[-1].gr = obj.gr
        # val[-1].t_grow = obj.t_grow
        # val[-1].vd = obj.vd

    if discr_time:
        nstep = temp_params['nstep']
        t_max = nstep * temp_params['dt'] * np.log(2)/temp_params['lambda']  # maximum time of the previous simulation, so that we know what time to
        # start our new cells at. Only use this if the previous simulation was a discretized time one.
        # Otherwise start at random.
        for i0 in range(len(val)):
            val[i0].t_div = temp[i0].t_div - t_max
    for i0 in range(len(val)):
        if val[i0].t_div < 0:
            raise ValueError('Error in starting_popn_seeded')
    return val


def division_rate_calculation(temp_cells, temp_params, temp_obs):
    dt = temp_params['num_rate_steps']*(temp_obs[1][1]-temp_obs[1][0])
    # this is the time interval over which we calculate the division rate
    div_ages = []
    cell_ages = []
    timepoints = range(len(temp_obs[1])-1-10*temp_params['num_rate_steps'], len(temp_obs[1])-1-temp_params['num_rate_steps'], temp_params['num_rate_steps'])
    for i0 in timepoints:
        div_ages.append([obj.t_grow for obj in temp_cells if
                         temp_obs[1][i0] < obj.t_div <= temp_obs[1][i0+temp_params['num_rate_steps']]])
        # this gives the collection of cells that divide in each timestep
        cell_ages.append([temp_obs[1][i0]-obj.tb for obj in temp_cells if obj.tb < temp_obs[1][i0] and
                         obj.t_div > temp_obs[1][i0]])
        # the ages of all cells at the start of this timepoint
    return div_ages, cell_ages, temp_obs[0][:len(temp_obs[1])-temp_params['num_rate_steps']], dt
