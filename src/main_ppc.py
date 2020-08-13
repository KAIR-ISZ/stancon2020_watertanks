import sys
import os

import pystan
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from lib import stan_utility
from scipy.interpolate import BSpline
from lib.DA_tools import ribbon_plot, get_quantiles
from lib.DA_colors import *
from lib.utility import  create_spline_matrix, read_data, pickle_stan_model, load_stan_model, fit_data, ppc_fit
import pickle


plt.style.context('seaborn-white')
mpl.rcParams['figure.dpi'] = 200

if __name__ == '__main__':
    """
    Summary:
    Script performs piore predictive computed for the selected trajectory. 
    As a result, we get ready-to-use models in the form of pickle files.

    
    Parameters:
    trajectories_name (str): name of the trajectory to carry out the analyzes. Possible names: 'h1', 'h2', 'h3'
    spl_order (int): the order of the interpolation used in the experiments
    num_knots (int): the order of the interpolation used in the experiments
    N (int): number of experimnets
    T (int): number of samples in experimnets
    
    Results:
    ppc_fit.pkl (file): results of fitting data for piore predictive computed
    """ 
        
    trajectories_name = 'h3'

    spl_order = 3
    num_knots = 7
    N = 20 
    T=1000


    wide_traj = pd.read_csv('trajectories_scen0.csv').iloc[:, 3::3]
    y_med = np.median(wide_traj.values.T,axis=0)
    centered = wide_traj-np.expand_dims(y_med,axis=1)
    centered['time']=[*range(0,T)]

    # number of trajectories
    trajectories0 = pd.wide_to_long(centered, stubnames=[trajectories_name], i='time', j='experiment', suffix='_\d+')
    trajectories0.index = trajectories0.index.droplevel(1)
    trajectories0 = trajectories0.reset_index().head(N*T)
    
    knot_list = np.quantile(trajectories0.time, np.linspace(0, 1, num_knots))
    knots = np.pad(knot_list, (spl_order, spl_order), mode="edge")
    B = BSpline(knots, np.identity(num_knots + 2),
                k=spl_order)(trajectories0.time.values[0:T])
    
    spline_fit2_ppc = stan_utility.compile_model('./models/spline_fit2_ppc.stan')
    
    if not os.path.isfile('./pickle/ppc_fit.pkl'):
        ppc = ppc_fit(spline_fit2_ppc, B, T, trajectories0[trajectories_name].values)
    
        pickle_stan_model('./pickle/ppc_fit.pkl', spline_fit2_ppc,
                          ppc=ppc)
    else:
        ppc_dic = load_stan_model('./pickle/ppc_fit.pkl')
        ppc = ppc_dic['ppc']
        
    y_pred = ppc.extract()['y_pred']
    beta = ppc.extract()['beta']
    mn_beta = np.mean(beta, axis=0)
    sd_beta = np.std(beta, axis=0)
    spline_components = B@np.diag(mn_beta)

    y_m = np.max(spline_components, axis=0)
    x_m = np.argmax(spline_components, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax2 = axes[0]

    for i in range(num_knots+2):
        ax2.plot(trajectories0.time.values[0:T],
                (spline_components[:, i]), color=DARK, zorder=0)
    
    ax2.errorbar(x_m, y_m, ls='none', yerr=2 *
                sd_beta[i]*y_m[i], color='black', capsize=4)
    ax2.text(0, 6, s='B-spline functions', color=DARK)
    ax2.set_ylabel('B-spline values')

    ax3 = axes[1]
    ax3 = ribbon_plot(trajectories0.time.values[0:T], y_pred, ax3, probs=[
                    2.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 97.5])
    ax3.set_ylabel('Water level')
    ax3.set_xlabel('Time')
    ax3.set_xticks([0, 500, 1000])
    ax3.set_xticklabels(['0 s', '500 s', '1000 s'])

    fig.tight_layout()
    fig.savefig('Spline_approx_healthy_ppc.png')

    plt.show()
