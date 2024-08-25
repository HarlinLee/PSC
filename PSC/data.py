import numpy as np
from pymanopt.manifolds.stiefel import Stiefel

def random_point(d, k, s=1):
    z = np.random.randn(s, d, k)
    if d < 200:
        u, _, vh = np.linalg.svd(z)     
        res = np.einsum('ijk,ikl->ijl', u[:, :, :k], vh) 
    else:
        res = []
        for z_i in z:
            u, _, vh = np.linalg.svd(z_i)  
            res.append(u[:, :k].dot(vh))
        res = np.array(res)
    if s == 1:
        res = res[0]
    return res

def rotation_mat(n):
    return random_point(n, n)

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


def sphere_point_cloud(N, n, s, epsilon=0, alpha=None):
    # Generates unifomly distributed points in neighborhood of subsphere

    if alpha is None:
        alpha = random_point(N, n)

    xs = random_point(n, 1, s)
    ys = np.matmul(alpha, xs)

    perturbations = np.array([random_tangent_vector_sphere(y) for y in ys])

    ys = ys + epsilon * perturbations

    ys = ys / np.sqrt((ys ** 2).sum(1))[..., np.newaxis]
    ys = ys.reshape(s, N, 1)

    return {'ys': ys, 'alpha': alpha, 'epsilon': epsilon, 'xs': xs}

def stiefel_point_cloud(N, n, k, s, epsilon=0, alpha=None):
    # If no initial point given, generate one
    if alpha is None:
        alpha = random_point(N, n)

    # Sample points in V_k(Rd) and map them into V_k(Rn)
    xs = random_point(n, k, s)
    ys = alpha.dot(xs).transpose((1,0,2))

    # project back to Stiefel
    ys = ys + epsilon*np.random.randn(s, N, k) # iid Gaussian coefficients N x k
    u, s, vh = np.linalg.svd(ys)
    ys = np.einsum('ijk,ikl->ijl', u[:, :, :k], vh)

    return {'ys':ys, 'alpha':alpha, 'epsilon':epsilon, 'xs':xs}

def stiefel_point_cloud_perp(N, n, k, s, epsilon=0, alpha=None):
    # If no initial point given, generate one
    if alpha is None:
        alpha = random_point(N, n)

    # Sample points in V_k(Rd) and map them into V_k(Rn)
    xs = random_point(n, k, s)
    ys = alpha.dot(xs).transpose((1,0,2))

    # project noise matrix to the subspace orthogonal to alpha
    P = np.eye(N)- alpha.dot(alpha.T)
    basis, _, _ = np.linalg.svd(P, full_matrices=True)

    orth_to_alpha = np.sum(np.abs(basis.T.dot(alpha)), axis=1) < 1e-5
    # or sort and get the smallest N-n
    basis = basis.T[orth_to_alpha].T # N x (N-n)

    assert(sum(orth_to_alpha) == N-n)

    noise_vectors = epsilon*np.random.randn(s, N-n, k) # iid Gaussian coefficients (N-n) x k
    noise_vectors = basis.dot(noise_vectors).transpose((1,0,2)) # linear combinations of the basis vector

    # project back to Stiefel
    ys = ys + noise_vectors
    u, s, vh = np.linalg.svd(ys)
    ys = np.einsum('ijk,ikl->ijl', u[:, :, :k], vh)

    return {'ys':ys, 'alpha':alpha, 'epsilon':epsilon, 'xs':xs}
