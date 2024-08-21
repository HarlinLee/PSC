import numpy as np
from .projections import pi_alpha_all

def check_orth(A):
    return np.allclose(A.T.dot(A), np.eye(A.shape[1]))

def dist_St(X, Y):
    return np.linalg.norm(X - Y, 'fro')

def dist_Gr(X, Y):
    return np.linalg.norm(X.dot(X.T) - Y.dot(Y.T), 'fro')

def projection_cost(alpha, ys):
    return np.sum((ys - pi_alpha_all(alpha, ys))**2)/len(ys)

def nuc_cost(alpha, ys):
    Y = alpha.T.dot(ys).transpose((1, 0, 2))
    u, s, vh = np.linalg.svd(Y)
    return -np.sum(s)/len(ys)

