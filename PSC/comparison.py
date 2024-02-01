from . import utils
from . import projections
import numpy as np
import autograd.numpy as anp
import pandas as pd

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA

from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.core.problem import Problem
from pymanopt.function import autograd, numpy
from pymanopt import optimizers

def PSC_points(points,d):
    #Project points to lower-dimensional Stiefel of d rows, k cols

    N=points[0].shape[0]
    alpha=Stiefel(N,d).random_point()
    found_alpha=projections.manopt_alpha(points,alpha)
    projected = projections.yhat_alpha_all(found_alpha, points)
    return projected, found_alpha


def PGA_tangent_vecs(points, d):
    #Project to the first d principal components
    s = points.shape[0]
    N = points.shape[1]
    st_kN = Hypersphere(N - 1)

    mean_estimate = frechetmean(points, N, 1).reshape(N, )  # Need shape (N,) for mean_estimate in TPCA fit

    tpca = TangentPCA(st_kN.metric, n_components=d)

    tpca.fit(points.reshape(s, N), base_point=mean_estimate)

    pc_matrix = np.array(tpca.components_[:d-1])

    pc_projection = np.matmul(pc_matrix.T, pc_matrix)

    tangent_data = st_kN.metric.log(points.reshape(s, N), mean_estimate)

    vector_tangent_data = tangent_data.reshape(s, 1, N)

    projected_tangent_data = np.matmul(pc_projection, vector_tangent_data.transpose(0, 2, 1))

    projected_tangent_data = projected_tangent_data.reshape(s, N)

    return [projected_tangent_data, mean_estimate]


def PGA_points(points, d):
    s = points.shape[0]
    N = points.shape[1]
    st_kN = Hypersphere(N - 1)
    PGA = PGA_tangent_vecs(points, d)
    return st_kN.metric.exp(PGA[0], PGA[1])

def var(points,frechet_mean):
    #Calculates variance from intrinsic Frechet mean using Riemannian distance
    d=0
    s=points.shape[0]
    N=points.shape[1]
    manifold=Hypersphere(N-1)
    for i in range(s):
      d += HypersphereMetric(manifold).dist(points[i].T,frechet_mean.T)
    return d.item()

def frechetmean(points,d,k):
  #Calculated Frechet mean for extrinsic coordinates
    if k==1:
      manifold=Hypersphere(d-1)
    else:
      manifold=Stiefel(d,k)
    mean = FrechetMean(manifold.metric, max_iter=100)
    mean.fit(points)
    return mean.estimate_.reshape(d,1)

def compare_PGA_var(sample,d):
    N=sample['N']
    n=sample['n']
    s=sample['s']
    epsilon=sample['epsilon']
    manifold=Hypersphere(N-1)

    mean=frechetmean(sample['points'].reshape(s,N),N,1)
    initial_variance = var(sample['points'],mean)

    projected=PGA_points(sample['points'].reshape(s,N),d)
    projected_mean=frechetmean(projected,N,1)

    projected=projected.reshape(s,N,1)
    projected_variance = var(projected, projected_mean)

    return projected_variance/initial_variance

def compare_PSC_var(sample,d):
    #Compare variance explained for PSC projection
    N=sample['N']
    n=sample['n']
    s=sample['s']
    alpha=sample['ground truth']
    epsilon=sample['epsilon']
    #manifold=Hypersphere(N-1)
    manifold=Hypersphere(d-1)

    mean=frechetmean(sample['points'].reshape(s,N),N,1)
    initial_variance = var(sample['points'],mean)

    projected, found_alpha = PSC_points(sample['points'],d)
    projected_mean = frechetmean(projected.reshape(s,d),d,1)

    projected=projected.reshape(s,d,1)
    projected_variance = var(projected, projected_mean)

    return projected_variance/initial_variance

def output_var_df(N, n, s, t, eps_vec, n_components):
    tmp = []
    for epsilon in eps_vec:
      for i in range(t):
        sample=utils.sphere_point_cloud(N,n,s,epsilon)
        for j in range(1,n_components+1):
          sample_df=pd.DataFrame(columns=['dims','PSC','PGA','epsilon','iteration'])
          sample_df['PGA'] =  [compare_PGA_var(sample,j+1)]
          sample_df['PSC'] = [compare_PSC_var(sample,j+1)]
          sample_df['dims'] = [j]
          sample_df['epsilon'] = [epsilon]
          sample_df['iteration'] = [i]
          tmp.append(sample_df)
    ratios_df=pd.concat(tmp,ignore_index=True)
    return ratios_df