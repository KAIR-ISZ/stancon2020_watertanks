import sys
import os

import pystan
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from lib import stan_utility
from lib.DA_tools import ribbon_plot, get_quantiles
from lib.DA_colors import *
from lib.utility import  create_spline_matrix, read_data, pickle_stan_model, load_stan_model, fit_data, ppc_fit
from scipy.interpolate import BSpline
import pickle

if __name__ == '__main__':
    """
    Summary:
    Script processes data from experiments describing the process of filling the tanks. 
    As a result, we get ready-to-use models in the form of pickle files
    
    Parameters:
    trajectories_name (str): name of the trajectory to carry out the analyzes. Possible names: 'h1', 'h2', 'h3'
    spl_order (int): the order of the interpolation used in the experiments
    num_knots (int): the order of the interpolation used in the experiments
    
    Results:
    simulated__h3.pkl (file): analysis results for simulation data
    experiment__h2.pkl (file): analysis results for experimental data
    """ 
    
    trajectories_name = 'h3'
    spl_order = 3
    num_knots = 15
       
    trajectories0, N, T = read_data('../data/result/result_med_scen0.csv')
    trajectories1, N, T = read_data('../data/result/result_med_scen1.csv')
    trajectories2, N, T = read_data('../data/result/result_med_scen2.csv')

    BN = create_spline_matrix(trajectories1, N,T,spl_order,num_knots)
    B0 = BN[0:T]

    model = stan_utility.compile_model('./models/spline_fit_4.stan')
    
    if not os.path.isfile('./pickle/simulated__h3.pkl'):
        fit4  = fit_data(model, BN, T, trajectories0[trajectories_name].values[0:N*T])
        fit4_ft1 = fit_data(model, B0, T, trajectories1[trajectories_name].values[0:T])
        fit4_ft2 = fit_data(model, B0, T, trajectories2[trajectories_name].values[0:T])
    
        pickle_stan_model('./pickle/simulated__h3.pkl', model, 
                    fit_healthy=fit4, 
                    fit_fault1=fit4_ft1,
                    fit_fault2=fit4_ft2)
    else:
        dict_ = load_stan_model('./pickle/simulated__h3.pkl')
        
    experiment, N, T = read_data('../data/result/result_med_600.csv')
    experiment1 = experiment[experiment.normal]
    experiment2 = experiment[~(experiment.normal)]
    
    N = 10
    BN = create_spline_matrix(experiment1, N, T, spl_order,num_knots)
    B0 = BN[0:T]
    
    if not os.path.isfile('./pickle/experiment__h2.pkl'):
        fit4  = fit_data(model, BN, T, experiment1[trajectories_name].values[0:N*T])
        fit4_ft1  = fit_data(model, B0, T, experiment1[trajectories_name].values[0:T])
        
        pickle_stan_model('./pickle/experiment__h2.pkl', model, 
                        fit_healthy=fit4, 
                        fit_fault1=fit4_ft1)
    else:
        dict2_ = load_stan_model('./pickle/experiment__h2.pkl')
        