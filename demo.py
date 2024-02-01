from PSC.projections import PCA, manopt_alpha, projection_cost, nuc_cost
from PSC.utils import get_samples

import numpy as np
from pymanopt.manifolds.stiefel import Stiefel
import matplotlib.pyplot as plt

N = 100
n = 10
k = 4


st_Nk = Stiefel(N, k)
ys = get_samples(st_Nk, 200)

print('shape of Y:', ys.shape)
print()

print('random alpha:')

alpha_random = Stiefel(N, n).random_point()

print('projection cost:', projection_cost(alpha_random, ys))
print('nuclear norm cost:', nuc_cost(alpha_random, ys))
print()

print('alpha PCA:')
alpha_PCA = PCA(ys, n)

print('projection cost:', projection_cost(alpha_PCA, ys))
print('nuclear norm cost:',nuc_cost(alpha_PCA, ys))
print()

print('alpha GD:')
alpha_GD = manopt_alpha(ys, alpha_PCA, verbosity=1)

print('projection cost:', projection_cost(alpha_GD, ys))
print('nuclear norm cost:',nuc_cost(alpha_GD, ys))