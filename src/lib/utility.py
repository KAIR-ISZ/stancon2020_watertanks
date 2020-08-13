import os
import pickle


import pandas as pd
import numpy as np

from scipy.interpolate import BSpline


def read_data(file_path:str) -> pd.DataFrame:
    if os.path.isfile(file_path):
        data = pd.read_csv(file_path)
        data['h3'] = data['h3'].apply(pd.to_numeric, errors='coerce')
        samples_length = len(data.time.unique())
        experiment_number = len(data['experiment_number'].unique())
        return data,experiment_number,samples_length
    else:
        raise ValueError('{0} don\'t exist.'.format(file_path))

def create_spline_matrix(data, N, T, spl_order=3, num_knots=7):
    knot_list = np.quantile(data.time, np.linspace(0, 1, num_knots))
    knots = np.pad(knot_list, (spl_order, spl_order), mode="edge")
    B = BSpline(knots, np.identity(num_knots + 2),
                k=spl_order)(data.time.values[0:T])
    return np.tile(B, (N, 1))

def pickle_stan_model(path:str, model, **kwargs):
    dir_path, file_name = os.path.split(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    kwargs['model'] = model 
    with open(path, "wb") as f:
        pickle.dump(kwargs, f, protocol=-1)

def load_stan_model(path:str):
    with open(path,'rb') as f:
            return pickle.load(f) 
        
def fit_data(model, B, length, sig, seed=6062020):
    data_fit = dict(N=B.shape[0],
                    K=B.shape[1],
                    L=length,
                    x=B,
                    y=sig)              
    return model.sampling(data=data_fit, seed=seed)

def ppc_fit(model, B, length, sig, 
            iter=1000, warmup=0, chains=1, refresh=1000,
            algorithm ='Fixed_param', seed=6062020):
    data_fit = dict(N=B.shape[0],
                    K=B.shape[1],
                    L=length,
                    x=B,
                    y=sig)              
    return model.sampling(data=data_fit, iter=iter, warmup=warmup, 
                           chains=chains, 
                           refresh=refresh,
                           algorithm=algorithm,
                           seed=seed)
    
def mahalanobis_depth(x,x_bar,invSigma):
    dev=np.expand_dims(x-x_bar,axis=1)
    MD=dev.T@invSigma@dev
    return np.asscalar(1/(1+MD))