
import sys

import stan_utility
import pystan
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from lib.DA_tools import ribbon_plot, get_quantiles
from lib.DA_colors import *
from scipy.interpolate import BSpline
import pickle



def create_spline_matrix(N,T,time,spl_order=3,num_knots=7):
    '''
    N - Number of time series,
    T - number of samples
    time - array/series of time values
    
    '''
    time=np.array(time) 
    knot_list = np.quantile(time, np.linspace(0, 1, num_knots))
    knots = np.pad(knot_list, (spl_order, spl_order), mode="edge")
    B = BSpline(knots, np.identity(num_knots + 2),
                k=spl_order)(time[0:T])
    # Design matrix
    return np.tile(B, (N, 1))

# number of trajectories for healthy model
N = 20 
# number of samples per trajectory
M=1000

trajectories1 = pd.read_csv('../data/result/result_med_scen0.csv')
trajectories2 = pd.read_csv('../data/result/result_med_scen1.csv')
trajectories3 = pd.read_csv('../data/result/result_med_scen2.csv')


spl_order = 3
num_knots = 15

BN = create_spline_matrix(N,M,trajectories1.time.values,spl_order,num_knots)
B0 = BN[0:M]

model = stan_utility.compile_model('spline_fit_4.stan')

data_fit4 = dict(N=BN.shape[0],
                 K=BN.shape[1],
                 L=M,
                 x=BN,
                 y=trajectories1.h3.values[0:N*M])

               
fit4 = model.sampling(data=data_fit4, seed=6062020)

data_fit4_ft1 = dict(N=B0.shape[0],
                 K=B0.shape[1],
                 L=M,
                 x=B0,
                 y=trajectories2.h3.values[0:M])

fit4_ft1 = model.sampling(data=data_fit4_ft1, seed=6062020)


data_fit4_ft2 = dict(N=B0.shape[0],
                 K=B0.shape[1],
                 L=M,
                 x=B0,
                 y=trajectories3.h3.values[0:M])

fit4_ft2 = model.sampling(data=data_fit4_ft2, seed=6062020)


with open("simulated__h3.pkl", "wb") as f:
    pickle.dump({'model' : model, 
                 'fit_healthy' : fit4, 
                 'fit_fault1':fit4_ft1,
                 'fit_fault2':fit4_ft2}, f, protocol=-1)






