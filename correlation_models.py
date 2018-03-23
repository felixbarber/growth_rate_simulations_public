import numpy as np

# this assumes model # 3 (i.e. asymmetric, non-budding, growth rate. Should approximately apply to adder model for small
# noise in interdivision times.


def fn(par1):  # <t_n+1 t_n>_c (independent of celltype n+1 provided n is unspecified)
    return (par1['alpha']*(1-par1['alpha'])*(np.log(par1['r'])/(2*par1['lambda']))**2-par1['alpha']*par1['td_std'][0]**2)/(2-par1['alpha'])


def fn_1(par1, celltype):  # sigma_tau^2
    if celltype ==0 or celltype == 1:
        temp = par1['alpha']**2*(mom2_x(par1,2)-av_x(par1, 2)**2)*(1-2*par1['alpha'])/par1['lambda']**2+par1['alpha']**2*fn_1(par1,2)+par1['td_std'][0]**2
    elif celltype == 2:
        temp = (2*par1['td_std'][0]**2+par1['alpha']*(np.log(par1['r'])/(2*par1['lambda']))**2)/(2-par1['alpha'])
    return temp


def av_x(par1, celltype):  #<x>
    b=par1['r']/(1+par1['r'])
    if celltype==0:
        temp = np.log(2*(1-b))+(1-par1['alpha'])*np.log(4*b*(1-b))/(2*par1['alpha'])
    if celltype == 1:
        temp = np.log(2*b)+(1-par1['alpha'])*np.log(4*b*(1-b))/(2*par1['alpha'])
    if celltype == 2:
        temp = np.log(4*b*(1-b))/(2*par1['alpha'])
    return temp


def mom2_x(par1, celltype): # <x^2>
    b=par1['r']/(1+par1['r'])
    if celltype==0:  #Mothers
        temp = np.log(2*(1-b))**2+2*(1-par1['alpha'])*np.log(2*(1-b))*av_x(par1, 2)+mom2_x(par1, 2)*(1-par1['alpha'])**2+(par1['lambda']*par1['td_std'][0])**2
    if celltype == 1:  # daughters
        temp = np.log(2*(b))**2+2*(1-par1['alpha'])*np.log(2*(b))*av_x(par1, 2)+mom2_x(par1, 2)*(1-par1['alpha'])**2+(par1['lambda']*par1['td_std'][0])**2
    if celltype == 2:
        temp = ((par1['lambda']*par1['td_std'][0])**2+(np.log(par1['r'])/2)**2)/(par1['alpha']*(2-par1['alpha']))+av_x(par1, 2)**2
    return temp


def pcc_md_diffgen(par1):
    temp = fn(par1)/(fn_1(par1, 1)**0.5*fn_1(par1, 2)**0.5)
    return temp


def pcc_mm_diffgen(par1):
    temp = fn(par1) / (fn_1(par1, 0) ** 0.5 * fn_1(par1, 2) ** 0.5)
    return temp


def pcc_md_samegen(par1):
    if par1['td_std'][0] == par1['td_std'][0]:
        temp = 1-par1['td_std'][0]**2/fn_1(par1, 1)
    else:
        raise ValueError('Correlation assumes distinct division time noise for mothers and daughters')
    return temp
