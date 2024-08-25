from PSC.projections import PCA, manopt_alpha
from PSC.utils import projection_cost, nuc_cost
from PSC.data import random_point, stiefel_point_cloud

import numpy as np
import matplotlib.pyplot as plt

N = 100
n = 100
k = 10
s = 200

ys = stiefel_point_cloud(N, n, k, s, 0.1)['ys']

print(f'Dimensions N={N}, n={n}, k={k}, number of samples {s}.')
print('shape of Y:', ys.shape)
print()

print('random alpha:')

alpha_random = random_point(N, n)

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
