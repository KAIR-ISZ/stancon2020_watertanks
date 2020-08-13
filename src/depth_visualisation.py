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
from lib.utility import  create_spline_matrix, read_data, pickle_stan_model, load_stan_model, fit_data, ppc_fit, mahalanobis_depth
from scipy.interpolate import BSpline
import pickle

if __name__ == '__main__':
    
    """
    Summary:
    Script allows you to analyze the depth of probability using Mahalanobis distance for simulation data.

    
    Parameters:
    trajectories_name (str): name of the trajectory to carry out the analyzes. Possible names: 'h1', 'h2', 'h3'
    spl_order (int): the order of the interpolation used in the experiments
    num_knots (int): the order of the interpolation used in the experiments
    stan_model_path (str): path to file with stan model used in analise process
    pickle_path (int): path to file with 
    """ 
    
    trajectories_name = 'h3'
    spl_order = 3
    num_knots = 15
    
    stan_model_path = './models/spline_fit_4.stan'
    pickle_path = './pickle/simulated__h3.pkl'
    
    model = stan_utility.compile_model(stan_model_path)
    fit_dict = load_stan_model(pickle_path)
    fit_healthy = fit_dict['fit_healthy']
    fit_fault1 = fit_dict['fit_fault1']
    fit_fault2 = fit_dict['fit_fault2']
    
    beta_healthy = fit_healthy.extract('beta').get('beta')
    beta_fault = fit_fault2.extract('beta').get('beta')
    y_healthy = fit_healthy.extract('y_pred').get('y_pred')
    y_pred_ft = fit_fault2.extract('y_pred').get('y_pred')
    
    mn_beta = np.mean(beta_healthy, axis=0)
    beta_cov = np.cov(beta_healthy.T)
    invSigma = np.linalg.inv(beta_cov)
    depths_healthy=[]

    for b in beta_healthy:
        depths_healthy.append(mahalanobis_depth(b,mn_beta,invSigma))
    depths_healthy = np.array(depths_healthy)
    
    depths_ft=[]

    for b in beta_fault:
        depths_ft.append(mahalanobis_depth(b,mn_beta,invSigma))
    depths_ft = np.array(depths_ft)
        
        
    fig, axes = plt.subplots(1, 1, figsize=(7, 4),sharex=True)
    ax1=axes
    ax1.hist(depths_healthy,bins=50,color=MID,edgecolor=MID_HIGHLIGHT, density=True)
    ax1.hist(depths_ft,bins=20,color=DARK,edgecolor=DARK, density=True)
    ax1.set_xscale('log')
    ax1.set_ylim([0, 30])
    ax1.set_yticks([])
    plt.show()
    
    N = 1 
    M=1000
    sd_beta = np.std(beta_healthy, axis=0)

    trajectories1 = pd.read_csv('../data/result/result_med_scen0.csv')
    trajectories2 = pd.read_csv('../data/result/result_med_scen1.csv')
    trajectories3 = pd.read_csv('../data/result/result_med_scen2.csv')

    B0 = create_spline_matrix(trajectories1,N,M,spl_order,num_knots)

    spline_components = B0@np.diag(mn_beta)
    y_m = 0*mn_beta
    for i in range(len(y_m)):
        y_m[i] = max(spline_components[:,i].min(), spline_components[:,i].max(), key=abs)
    x_m = np.argmax(np.abs(spline_components), axis=0)

    fig, ax_spl = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    # #ax2.set_ylim((-1, 14))
    for i in range(num_knots+2):
        ax_spl.plot(trajectories1.time.values[0:1000],(spline_components[:, i]), color=DARK, zorder=0)
        ax_spl.errorbar(x_m[i], y_m[i], ls='none', yerr=2 * sd_beta[i]*y_m[i], color='black', capsize=4)
    ax_spl.set_ylabel('B-spline values')

    plt.show()
    
    fig, axes2 = plt.subplots(2, 2, figsize=(16, 9))
    ax_spl=axes2[0,0]
    for i in range(num_knots+2):
        ax_spl.plot(trajectories1.time.values[0:1000],(spline_components[:, i]), color=DARK, zorder=0)
        ax_spl.errorbar(x_m[i], y_m[i], ls='none', yerr=2 * sd_beta[i]*y_m[i], color='black', capsize=4)
    ax_spl.set_ylabel('B-spline values')
    ax_spl.set_title('B-splines and uncertainty')
    ax_y=axes2[0,1]
    ax_y = ribbon_plot(trajectories1.time.values[0:1000], y_healthy, ax_y)
    qs = get_quantiles(trajectories1.pivot(index='time',columns='experiment_number',values=trajectories_name).values.T, [2.5, 50, 97.5])
    ax_y.plot(trajectories1.time.values[0:1000],
            qs[0, :], color='black', linestyle='--')
    ax_y.plot(trajectories1.time.values[0:1000],
            qs[2, :], color='black', linestyle='--')
    ax_y.set_ylabel('Water level deviation')
    ax_y.set_xlabel('Time')
    ax_y.set_xticks([0, 500, 1000])
    ax_y.set_xticklabels(['0 s', '500 s', '1000 s'])
    ax_y.set_title('Posterior predictive sim. of healthy state')


    ax_y_ft=axes2[1,0]
    ax_y_ft = ribbon_plot(trajectories2.time.values[0:1000], y_pred_ft, ax_y_ft)
    ax_y_ft.scatter(trajectories2.time.values[0:1000],trajectories3[trajectories_name].values[0:1000],color='black',s=6)
    ax_y_ft.set_ylabel('Water level deviation')
    ax_y_ft.set_xlabel('Time')
    ax_y_ft.set_xticks([0, 500, 1000])
    ax_y_ft.set_xticklabels(['0 s', '500 s', '1000 s'])
    ax_y_ft.set_title('Posterior predictive sim. of faulty state')


    ax_dpth=axes2[1,1]
    ax_dpth.hist(depths_healthy,bins=50,color=MID,edgecolor=MID_HIGHLIGHT, density=True)
    ax_dpth.hist(depths_ft,bins=20,color=DARK,edgecolor=DARK, density=True)
    ax_dpth.set_xscale('log')
    ax_dpth.set_ylim([0, 30])
    ax_dpth.set_yticks([])
    ax_dpth.set_title('Data depth')

    fig.tight_layout()
    plt.show()
        

