import numpy as np
import cvxpy as cp
from typing import Tuple
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

def scale_and_discretize(X:np.ndarray, disc_strategy:str, disc_bins:int) -> Tuple:
    '''
    Scale and discretize  an NxD numpy array for use in D-optimal experiment design

    parameters
    ----------
        X : np.ndarray
            - Dataset with observations as rows and parameters as columns
        disc_strategy : str
            - Discretization strategies for the data; see KBinsDiscretizer documentaiton
            - One of {'uniform', 'quantile', 'kmeans'}
        disc_bins : int
            - Number of bins to discretize each variable into

    returns
    ----------
        Xt: np.ndarray
            - Scaled and transformed data
        (Scale, Variance): Tuple(np.ndarray)
            - Tuple of mean and variance vectors from normalization
        Bin edges: List(np.ndarray)
            - Bin edges from discretization

    '''
    # scalarize
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    
    # discretize
    discretizer = KBinsDiscretizer(n_bins=disc_bins, encode='ordinal', strategy=disc_strategy)
    Xt = discretizer.fit_transform(Xs)
    
    return Xt, (scaler.scale_, scaler.var_), discretizer.bin_edges_



def doptimal(X:np.ndarray, m:int) -> Tuple:
    '''Run D-optimal sample design for normalized and discretized matrix X

    parameters
    ----------
        X : np.ndarray
            - normalized and discretized dataset from which to select experiments
        m : int
            - number of "relaxed" experiments to target; relaxed because the result
              will yield partial experiments; heuristics must be used to settle on 
              final outcome

    returns
    ----------
        V : np.ndarray
            - matrix of "experiments" to be considered in optimal solution
            - experiemtns as rows, experiment dimensions as columns
        l : np.ndarray
            - optimization variable; used to identify indices of optimal experiments
        stats: dict
            - dictionary of optimization statistics
    '''

    # setup
    V = np.unique(X, axis=0)
    p = len(V)
    l = cp.Variable((p,1))
    
    # problem
    obj = cp.Minimize(-cp.log_det(V.T @ cp.multiply(l, V)))
    constraints = [l >= 0, l <= 1]
    constraints += [cp.norm(l, 1) <= m]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
    except:
        prob.solve(solver='SCS')
    
    stats = {'status':prob.status, 
             'value':prob.value,
             'time':prob._compilation_time+prob._solve_time,
             'time_comp':prob._compilation_time,
             'time_solve':prob._solve_time}
    
    return V, l.value, stats


def get_indices(X:np.ndarray, V:np.ndarray, l:np.ndarray, m:int) -> np.array:
    '''
    helper function to get indices from output of d-optimal experiment design

    parameters
    ----------
        X : np.ndarray
            - normalized and discretized dataset from which to select experiments
        V : np.ndarray
            - experiemtns vector returned from D-optimal design
        l : np.ndarray
            - selector array returned from D-optimal design
        M : int
            - number of desired samples

    returns
    ----------
        idxs : np.array
         - Indices of original X matrix to select

    '''
    # vsub = V[(l >= cutoff).flatten()]
    # __, idxs = np.where((X==vsub[:,None]).all(-1))
    idxs = np.argsort(-l.flatten())[:m]
    return idxs