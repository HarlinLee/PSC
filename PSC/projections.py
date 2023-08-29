import numpy as np
import autograd.numpy as anp
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.core.problem import Problem
from pymanopt.function import autograd, numpy
from pymanopt import optimizers

def orth_proj(X, k=None):
    u, _, vh = np.linalg.svd(X)
    if not k:
        k = len(vh)
    return u[:, :k].dot(vh)

def yhat_alpha(alpha, y):   
    return orth_proj(alpha.T.dot(y))

def yhat_alpha_all(alpha, ys):   
    return np.array([yhat_alpha(alpha, y) for y in ys])

def pi_alpha(alpha, y):   
    return alpha.dot(yhat_alpha(alpha, y))

def pi_alpha_all(alpha, ys):   
    return np.array([pi_alpha(alpha, y) for y in ys])

def PCA(ys, n):
    m = len(ys)
    
    # sample covariance matrix
    S_hat = np.sum(np.array([y.dot(y.T) for y in ys]), axis=0)/m
    w, v = np.linalg.eigh(S_hat)

    return v[:, -n:]

def manopt_alpha(ys, alpha_init):        
    N, n = alpha_init.shape
    st_Nn = Stiefel(N, n) 

    @autograd(st_Nn)
    def cost(point):
        return -anp.sum([anp.linalg.norm(anp.dot(point.T, y), 'nuc') for y in ys])/len(ys)

    problem = Problem(st_Nn, cost=cost)
    optimizer = optimizers.SteepestDescent(verbosity=1)
    res = optimizer.run(problem, initial_point=alpha_init).point
    
    # print('nuc_cost of initial alpha', cost(alpha_init), 'nuc_cost of final alpha', cost(res))
    # print('projection_cost of initial alpha', projection_cost(alpha_init, ys), 'projection_cost of final alpha', projection_cost(res, ys))
    return res
