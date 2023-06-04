import numpy as np

def kernel(xi:np.array, xj:np.array, l:float) -> float:
    '''define gaussian process kernel; l is hyperparameter to tune'''
    return np.exp((-1/(2*np.square(l)))*np.dot(xi - xj, xi - xj))

def K(Xk:np.ndarray, l:float) -> np.ndarray:
    '''
    define FDD covariance matrix from gaussian process
    l is hyperparameter to tune for kernel function
    '''
    K = np.array([np.diagonal((Xk[i] - Xk) @ (Xk[i] - Xk).T) for i in range(len(Xk))])
    K = np.exp(-0.5*(1/l**2)*K)
    return K

def kx(x, Xk, l):
    '''define gaussian process kernel vector for given x and fixed Xk'''
    return np.array([kernel(x, xk, l) for xk in Xk])

def gp_mean(x, Xk, Kinv, fk, l):
    '''
    get mean of normally distributed f | Xk, fk
    pass Kinv to reduce need to invert matrix in this function
    '''
    return kx(x, Xk, l) @ Kinv @ fk

def gp_var(x, Xk, Kinv, l):
    '''
    get mean of normally distributed f | Xk, fk
    pass Kinv to reduce need to invert matrix in this function
    '''
    return kernel(x, x, l) - kx(x, Xk, l).T @ Kinv @ kx(x, Xk, l)