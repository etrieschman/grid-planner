import numpy as np
import cvxpy as cp

from utils_actlearn.gauss import *
from utils_actlearn.mes import f_obj

def gradf_obj(x, Xk, Kinv, l):
    '''gradient of MES objective (reformulated in write-up)'''
    k, d = Xk.shape
    gradf = np.zeros(d)
    for i in range(k):
        for j in range(k):
            grad_ij = (-2/l**2)*(x - 0.5*(Xk[i] + Xk[j]))
            grad_ij *= Kinv[i,j]*kernel(x, Xk[i], l)*kernel(x,Xk[j],l)
            gradf += grad_ij
    return gradf


def hessf_obj(x, Xk, Kinv, l):
    '''hessian of MES objective, reformulated in write-up'''
    k, d = Xk.shape
    hessf = np.zeros((d,d))
    for i in range(k):
        for j in range(k):
            hess_ij = (4/l**2)*np.outer((x - (1/2)*(Xk[i] + Xk[j])), x - (1/2)*(Xk[i] + Xk[j])) - 2*np.identity(d)
            hess_ij *= Kinv[i,j]*kernel(x, Xk[i], l)*kernel(x,Xk[j], l)
            hessf +=  hess_ij

    return hessf


def proj_hess(H):
    '''projection of a matrix onto the real-PSD cone'''
    lam, Q = np.linalg.eig(H)
    idx_poslam = np.where(lam > 0)[0]
    projH = Q[idx_poslam].T.real @ np.diag(lam[idx_poslam].real) @ Q[idx_poslam].real
    return projH



def fhat_obj_taylor(x:cp.Variable, xk:np.array, Xk:np.array, Kinv:np.array, l:float):
    '''Taylor appoximation of objective function'''
    P = proj_hess(hessf_obj(xk, Xk, Kinv, l))
    delta = x - xk
    return f_obj(xk, Xk, Kinv, l) + gradf_obj(xk, Xk, Kinv, l)@delta + cp.quad_form(delta, P, assume_PSD=True)



def scp_step_taylor(xk, Xk, Kinv, rho, Xlim, l):
    '''
    stochastic convex programming step centered at xk
    rho and Xlim define the SCP trust region, T^k
    '''
    k, d = Xk.shape
    x = cp.Variable(d)
    constraints = [x <= Xlim[1]]
    constraints += [x >= Xlim[0]]
    constraints += [cp.abs(x - xk) <= rho]

    prob = cp.Problem(cp.Minimize(fhat_obj_taylor(x, xk, Xk, Kinv, l)), constraints)
    prob.solve(solver='SCS')
    return prob.value, x.value



def scp_taylor(Xk, Kinv, x0, rho0, Xlim, l, alpha=0.1, beta=[1.1, 0.5], num_iters=50):
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
        __, xkp = scp_step_taylor(xk, Xk, Kinv, rho, Xlim, l)
    
        # update rho
        dec_pred = f_obj(xk, Xk, Kinv, l) - fhat_obj_taylor(xkp, xk, Xk, Kinv, l).value
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