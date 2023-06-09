import numpy as np
import timeit
from tqdm import tqdm

from utils_actlearn.gauss import *
from utils import dict_list, get_stats


def f_obj(x, Xk, Kinv, l):
    '''this is the true objective we want to minimize in MES'''
    return kx(x, Xk, l).T @ Kinv @ kx(x, Xk, l)

def update_datasets(dk, dleft, idxs):
    '''
    helper function to update datasets
    function takes index, idx, and moves it from dleft to dk'''
    if dk is None:
        dk = dleft[idxs]
    else:
        dk = np.vstack([dk, dleft[idxs]])
    dleft = np.delete(dleft, idxs, axis=0)
    return dk, dleft


def mes(scp_opt, X:np.ndarray, f:np.array, N:int, n0:int=2, 
        l:float=1., scp_param_override:dict=None):
    '''
    Use Maximum entropy search with SCP to train a Gaussian Process surrogate model on (X,f)
    In an applicaiton setting, we would not have full dataset f. instead, in the update
    datasets step, we would run the model for selected xample, Xleft[xk_idx]

    Parameters:
        - scp_opt : method of desired sequential convex optimization approach
        - X : dataset of encoded samples
        - f : dataset of model outputs (need to sequentiall generate in a real setting)
        - N : Total desired "model runs" to train the surrogate model
        - n0 : number of randomly selected sample points to initialize the Gaussian Process
        - l : key hyperparameter to tune for the Gaussian Process model
        - scp_param_override : dictionary to override scp_opt parameters
            - rho0 : initial trust region scaling of Xlim for SCP
            - alpha : condition to test for trust-region updates in SCP
            - beta : scaling parameters for trust-region updates
            - num_iters : number of SCP iterations

    Returns:
        - fpreds : a dictionary of arrays, where each array is the distribution generated
                   from the surrogate model
    '''
    # initialize logs
    fpreds = {}

    # initialize datasets
    idx_init = np.random.randint(0, len(X), n0)
    Xk, Xleft = update_datasets(None, X, idx_init)
    fk, fleft = update_datasets(None, f.reshape(-1,1), idx_init)
    fk, fleft = fk.flatten(), fleft.flatten()
    try:
        Kinv = np.linalg.inv(K(Xk, l))
    except:
        Kinv = np.linalg.pinv(K(Xk, l))

    # SCP parameters
    scp_params = {'rho0':0.01, 'alpha':0.1, 'beta':[1.1,0.5], 'num_iters':10}
    scp_params['Xlim'] = X.min(axis=0), X.max(axis=0)
    scp_params['x0'] = X.mean(axis=0)
    if scp_param_override is not None:
        for k, v in scp_param_override.items():
            scp_params[k] = v

    time_start = timeit.default_timer()
    runtimes = []
    for k in tqdm(range(n0, N), disable=True):
        # choose next sample
        if scp_opt == 'random':
            xk_idx = np.random.randint(0, len(Xleft))
        elif scp_opt == 'brute':
            xk_idx = np.argmin(np.array([f_obj(x, Xk, Kinv, l) for x in Xleft]))
        else:
            xk, log = scp_opt(Xk, Kinv, l=l, **scp_params)
            xk_idx = np.argmin(np.linalg.norm(Xleft - xk, ord=2, axis=1))            

        # update sample datasets
        Xk, Xleft = update_datasets(Xk, Xleft, xk_idx)
        fk, fleft = update_datasets(fk.reshape(-1,1), f.reshape(-1,1), xk_idx)
        fk, fleft = fk.flatten(), fleft.flatten()

        # fit model and run surrogate model for all samples
        # in case the matrix is singular, take the pseudo inverse
        try:
            Kinv = np.linalg.inv(K(Xk, l))
        except:
            Kinv = np.linalg.pinv(K(Xk, l))
        fpred = np.array([gp_mean(x, Xk, Kinv, fk, l) for x in X])
        runtimes += [timeit.default_timer() - time_start]
        fpreds[k] = fpred
        
    return fpreds, runtimes



def bootstrap_mes_stats(scp_opt, X, f, N, n0, l, n_bootstrap, stats):
    bootstrap_results = dict_list()

    for i in tqdm(range(n_bootstrap)):
        fpreds, runtimes = mes(scp_opt, X, f, N, n0, l)
        results = get_stats(f, fpreds)
        results['runtime'] = runtimes
        
        for stat in stats:
            bootstrap_results.add(stat, results[stat])

    
    for stat in stats:
        bootstrap_results[stat] = np.array(bootstrap_results[stat])
    
    return bootstrap_results, results, fpreds
