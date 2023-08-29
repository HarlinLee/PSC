import numpy as np
from projections import pi_alpha

from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.core.problem import Problem
from pymanopt import optimizers

import geomstats
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.hypersphere import HypersphereMetric

import matplotlib.pyplot as plt

def get_samples(st, p):
    return np.array([st.random_point() for i in range(p)])

def check_orth(A):
    return np.allclose(A.T.dot(A), np.eye(A.shape[1]))


def dist_St(X, Y):
    return np.linalg.norm(X - Y, 'fro')


def dist_Gr(X, Y):
    return np.linalg.norm(X.dot(X.T) - Y.dot(Y.T), 'fro')


def rotation_mat(n):
    return Stiefel(n, n).random_point()

def projection_cost(alpha, ys):
    return np.sum([np.linalg.norm(y - pi_alpha(alpha, y), 'fro')**2 for y in ys])/len(ys)

def nuc_cost(alpha, ys):
    return -np.sum([np.linalg.norm(alpha.T.dot(y), 'nuc') for y in ys])/len(ys)


def random_tangent_vector_sphere(point, as_columns=True):
    # Gives random unit tangent vector a given point on sphere (extrinsic)
    N = len(point) - 1
    point = point.reshape(N + 1, 1)
    projection = point.dot(point.T)
    basis, _, _ = np.linalg.svd(projection, full_matrices=True)
    embedding = np.array(
        [vect for vect in basis.T if abs(np.dot(vect, point)) < 0.05])  # Pick only the columns orthogonal to y
    random_vec = np.random.randn(N, 1)
    tangent_perturbation = np.dot(embedding.T, random_vec)  # Embed random vector in tangent space
    if as_columns != False:
        return tangent_perturbation / np.linalg.norm(tangent_perturbation)
    else:
        return tangent_perturbation.T / np.linalg.norm(tangent_perturbation)


def sphere_point_cloud(N, n, s, epsilon, alpha=None):
    # Generates unifomly distributed points in neighborhood of subsphere
    st_Nn = Stiefel(N, n)
    st_kn = Hypersphere(n - 1)
    st_kN = Hypersphere(N - 1)

    if alpha is None:
        alpha = st_Nn.random_point()

    points = st_kn.random_uniform(n_samples=s)
    pointsarray = np.asarray(points).reshape(s, n, 1)
    initial_ys = np.matmul(alpha, pointsarray)

    perturbations = np.array([random_tangent_vector_sphere(y) for y in initial_ys])

    perturbed_ys = initial_ys + epsilon * perturbations

    projected_ys = perturbed_ys / np.sqrt((perturbed_ys ** 2).sum(1))[..., np.newaxis]
    ys = projected_ys.reshape(s, N, 1)

    sample_result = {'N': N, 'n': n, 's': s, 'epsilon': epsilon, 'ground truth': alpha, 'xs': pointsarray, 'points': ys}

    return sample_result

