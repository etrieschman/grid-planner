import numpy as np
import cvxpy as cp

from utils_actlearn.mes import f_obj


def fhat_obj_particle(x, xk:np.array, P, q, r):
    '''particle method objective function approximation'''
    delta = x - xk
    obj = cp.quad_form(delta, P, assume_PSD=True) + q@delta + r
    return obj

def fit_quadratic(xk:np.array, Xk:np.ndarray, Kinv:np.ndarray, l:float, rho:float, num_samples:int):
    '''
    fit a quadratic function sample points evaluated with the surrogate model 
    within the trust region
    Used to get the particle method approxiimation to the objective function
    '''
    # sample data
    __, d = Xk.shape
    samples = np.array([np.random.uniform(low=xk-rho, high=xk+rho) for i in range(num_samples)])

    # fit quadratic
    P = cp.Variable((d,d), PSD=True)
    q = cp.Variable(d)
    r = cp.Variable()
    constraints = [P >> 0]

    obj = 0.0
    for sample in samples:
        obj += cp.square(fhat_obj_particle(sample, xk, P, q, r) - f_obj(sample, Xk, Kinv, l))
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver='SCS')
    
    return P.value, q.value, r.value


# optimization
def scp_step_particle(xk, Xk, Kinv, l, rho, Xlim, num_samples=50):
    '''
    stochastic convex programming step centered at xk
    rho and Xlim define the SCP trust region, T^k
    '''
    __, d = Xk.shape
    x = cp.Variable(d)
    constraints = [x <= Xlim[1]]
    constraints += [x >= Xlim[0]]
    constraints += [cp.abs(x - xk) <= rho]
    
    P, q, r = fit_quadratic(xk, Xk, Kinv, l, rho, num_samples)
    prob = cp.Problem(cp.Minimize(fhat_obj_particle(x, xk, P, q, r)), constraints)
    prob.solve(solver='SCS')
    
    return prob.value, x.value, (P, q, r)

def scp_particle(Xk, Kinv, x0, rho0, Xlim, l, alpha=0.1, beta=[1.1, 0.5], num_iters=50):
    '''
    full stochastic convex programming optimization for the next sample point, xk
    Code includes trust region (rho, Xlim) updates based on parameters alpha, beta
    '''
    # initialize
    xk = x0
    rho = rho0
    f_objs, dec_preds, decs = [], [], []
    
    # sequentially optimize
    for i in range(num_iters):
        __, xkp, (P, q, r) = scp_step_particle(xk, Xk, Kinv, l, rho, Xlim)
    
        # update rho
        dec_pred = f_obj(xk, Xk, Kinv, l) - fhat_obj_particle(xkp, xk, P, q, r).value
        dec = f_obj(xk, Xk, Kinv, l) - f_obj(xkp, Xk, Kinv, l)
        # log stuff
        f_objs += [f_obj(xk, Xk, Kinv, l)]
        dec_preds += [dec_pred]
        decs += [dec]

        # update rho
        if dec >= alpha*dec_pred:
            rho *= beta[0]
            xk = xkp
        else:
            rho *= beta[1]
        
    return xk, (f_objs, dec_preds, decs)