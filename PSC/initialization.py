import numpy as np
from .projections import PCA, pi_alpha_all
from .utils import dist_St

def ransac(ys_original, n, p = 0.99, tau = 3, max_iter = None, verbose=False):
    if max_iter is None:
        max_iter = len(ys_original)
        
    ys_ind = np.arange(len(ys_original))
    ys = ys_original.copy()
    
    for iter in range(max_iter):
        ind = np.random.choice(ys_ind, int(p*len(ys_ind)), replace=False)
        ysub = ys[ind]
        
        alpha_PCA = PCA(ysub, n)
        piys = pi_alpha_all(alpha_PCA, ysub)
        
        ds = []
        for y1, y2 in zip(ysub, piys):
            ds.append(dist_St(y1, y2))
        
        ds = np.array(ds)
        B = ind[ds - np.mean(ds) > np.std(ds)*tau] # indices to be removed
        if verbose:
            print(len(B), B)
        
        if len(B) == 0: # if B is empty
            break
            
        ys_ind = ys_ind[~np.isin(ys_ind, B)] # remove B
        
    return ys_original[ys_ind]

def tube_det(ys, alpha):
    # ys = s x N x k
    s, N, k = ys.shape
    
    if N < 200: 
        yTaaT = ys.swapaxes(2,1).dot(alpha).dot(alpha.T) # s x k x N
        A = np.einsum('ijk,ikl->ijl', yTaaT, ys) # s x k x k  
        keep = (np.linalg.det(A) > 1e-15)   
    else:
        aaT = alpha.dot(alpha.T)
        keep = []

        for y in ys:
            A = y.T.dot(aaT).dot(y)
            keep.append(np.linalg.det(A) > 1e-15)
        keep = np.array(keep)
    
    if sum(keep) < len(ys):
        print('removed', sum(~keep), 'points')
    return ys[keep]

def tube_rank(ys, alpha):
    s, N, k = ys.shape
    
    if N < 200:
        aTy = alpha.T.dot(ys).transpose((1, 0, 2))
        keep = (np.linalg.matrix_rank(aTy, tol=1e-15)==k)
    else:
        keep = []
        for y in ys:
            aTy = alpha.T.dot(y)
            r = np.linalg.matrix_rank(aTy, tol=1e-15)
            keep.append(r==k)
        keep = np.array(keep)
        
    if sum(keep) < len(ys):
        print('removed', sum(~keep), 'points')
    return ys[keep]
