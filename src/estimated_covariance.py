import sys
from lib import stan_utility
import pystan
import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.interpolate import BSpline
from datetime import datetime
import pickle
import os

def read_data(path:str) -> pd.DataFrame:
    if os.path.isfile(path):
        data = pd.read_csv(path)
        data['h3'] = data['h3'].apply(pd.to_numeric, errors='coerce')
        return data
    else:
        raise ValueError('{0} don\'t exist.'.format(path))
    
def create_spline_matrix(N,T,data,spl_order=3,num_knots=7):
    knot_list = np.quantile(data.time, np.linspace(0, 1, num_knots))
    knots = np.pad(knot_list, (spl_order, spl_order), mode="edge")
    B = BSpline(knots, np.identity(num_knots + 2),
                k=spl_order)(data.time.values[0:T])
    # Design matrix
    return np.tile(B, (N, 1))
    

if __name__ == "__main__":
    raw_trajectories = read_data('../data/result/result_600.csv') 
    T = len(raw_trajectories.time)
    trajectories = raw_trajectories[raw_trajectories['normal'] == True ]
    N = len(trajectories['experiment_number'].unique())
    trajectories = trajectories[['time','h1','h2','h3']]
    B = create_spline_matrix(N,T, trajectories)

    #kompilacja modelu (i przypsanie go do ziennej)
    spline_fit2 = stan_utility.compile_model('spline_fit2.stan')
    spline_fit2 = pystan.StanModel('spline_fit2.stan')
    pickle.dump( spline_fit2, open( "stanmodel_gaussian_healthy.pkl", "wb" ) )

    # dane do modelu
    data_fit2 = dict(N=B.shape[0],
                K=B.shape[1],
                L=T,
                x=B,
                y=trajectories.h3.values)
    # fitowanie
    fit2 = spline_fit2.sampling(data=data_fit2, seed=6062020)

    pickle.dump( fit2, open( "fit_gaussian_healthy.pkl", "wb" ) )

    y_pred = fit2.extract()['y_pred']
    beta = fit2.extract()['beta']
    pickle.dump( beta, open( "beta_coeffs_gaussian_healthy.pkl", "wb" ) )

    # srednie i odchylenia standardowe (brzegowe dla wspolczynnikow)
    mn_beta = np.mean(beta, axis=0)
    sd_beta = np.std(beta, axis=0)

    # Computation of mean splines
    B0 = B[0:T]

    spline_components = B0@np.diag(mn_beta)
    y_m = np.max(spline_components, axis=0)
    x_m = np.argmax(spline_components, axis=0)

    beta_cov = np.cov(beta.T)
    pickle.dump( beta_cov, open( "beta_coeffs_covariance_gaussian_healthy.pkl", "wb" ) )