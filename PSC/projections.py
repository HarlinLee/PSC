import numpy as np
import autograd.numpy as anp
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.core.problem import Problem
from pymanopt.function import autograd
from pymanopt import optimizers
from numpy.linalg import LinAlgError

def yhat_alpha_all(alpha, ys):
    Y = alpha.T.dot(ys).transpose((1, 0, 2)) # faster version of np.array([alpha.T.dot(y) for y in ys])
    u, s, vh = np.linalg.svd(Y) # faster version of np.array([np.linalg.svd(y) for y in Y])
    k = vh.shape[-1]
    return np.einsum('ijk,ikl->ijl', u[:, :, :k], vh) # faster version of np.array([u[i, :, :k].dot(vh[i, :, :]) for i in range(len(u))])

def pi_alpha_all(alpha, ys):
    yhats = yhat_alpha_all(alpha, ys)
    return alpha.dot(yhats).transpose((1,0,2))

def PCA(ys, n):
    S_hat = np.concatenate(ys, axis=-1) #np.sum(np.array([y.dot(y.T) for y in ys]), axis=0)/m
    u, _, _ = np.linalg.svd(S_hat, full_matrices=False)

    return u[:, :n]

def manopt_alpha(ys, alpha_init, verbosity=1):
    N, n = alpha_init.shape
    st_Nn = Stiefel(N, n)

    if N > 200:  
        @autograd(st_Nn)
        def cost(point):
            return -anp.sum([anp.linalg.norm(anp.dot(point.T, y), 'nuc') for y in ys])/len(ys)
    else:
        try:
            @autograd(st_Nn)
            def cost(point):
                Y = anp.dot(anp.transpose(point), ys)
                Y = anp.swapaxes(Y, 1, 0)
                u, s, vh = anp.linalg.svd(Y, full_matrices=False) # Seems slow for large N
                return -anp.sum(s)/len(ys)
                
        except LinAlgError as e:
            @autograd(st_Nn)
            def cost(point):
                return -anp.sum([anp.linalg.norm(anp.dot(point.T, y), 'nuc') for y in ys])/len(ys)

        
    problem = Problem(st_Nn, cost=cost)
    optimizer = optimizers.SteepestDescent(verbosity=verbosity)
    res = optimizer.run(problem, initial_point=alpha_init).point
    
    return res
