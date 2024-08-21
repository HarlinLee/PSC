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


def sphere_point_cloud(N, n, s, epsilon, alpha=None):

    # Generates unifomly distributed points in neighborhood of subsphere

    if alpha is None:
        alpha = random_point(N, n)

    points = random_point(n, 1, s)
    pointsarray = np.asarray(points).reshape(s, n, 1)
    initial_ys = np.matmul(alpha, pointsarray)

    perturbations = np.array([random_tangent_vector_sphere(y) for y in initial_ys])

    perturbed_ys = initial_ys + epsilon * perturbations

    projected_ys = perturbed_ys / np.sqrt((perturbed_ys ** 2).sum(1))[..., np.newaxis]
    ys = projected_ys.reshape(s, N, 1)

    sample_result = {'N': N, 'n': n, 's': s, 'epsilon': epsilon, 'ground truth': alpha, 'xs': pointsarray, 'points': ys}

    return sample_result

def stiefel_point_cloud(N, n, k, s, epsilon, alpha=None):
    # If no initial point given, generate one
    if alpha is None:
        alpha = random_point(N, n)

    # Sample points in V_k(Rd) and map them into V_k(Rn)
    xs = random_point(n, k, s)
    initial_ys = alpha.dot(xs).transpose((1,0,2))

    # project noise matrix to the subspace orthogonal to alpha
    P = np.eye(N)- alpha.dot(alpha.T)
    basis, _, _ = np.linalg.svd(P, full_matrices=True)

    orth_to_alpha = np.sum(np.abs(basis.T.dot(alpha)), axis=1) < 0.01
    # or sort and get the smallest N-n
    basis = basis.T[orth_to_alpha].T # N x (N-n)

    assert(sum(orth_to_alpha) == N-n)

    noise_vectors = epsilon*np.random.randn(s, N-n, k) # iid Gaussian coefficients (N-n) x k
    noise_vectors = basis.dot(noise_vectors).transpose((1,0,2)) # linear combinations of the basis vector

    # project back to Stiefel
    ys = initial_ys + noise_vectors
    u, s, vh = np.linalg.svd(ys)
    ys = np.einsum('ijk,ikl->ijl', u[:, :, :k], vh)

    # Create dictionary with Manopt output and initial data, e.g. y_is, choice of alpha, etc.
    sample_result = {'points':ys, 'alpha':alpha, 'epsilon':epsilon, 'xs':xs}

    return sample_result

# def stiefel_point_cloud(N,n,k,m,epsilon,alpha=None):
#     """
#     Given inputs, generates a list of points in a neighborhood of ground truth $\alpha$
#     :param N: Larger ambient dimension
#     :param n: Smaller dimension we aim to project to
#     :param k: Columns
#     :param m: Number of data points
#     :param epsilon: Optional parameter, initial point in St(d,n) to begin gradient descent
#     :param alpha: Optional parameter, ground truth as array. If not provided, one is randomly generated
#     :return: dictionary with 3 keys: points, alpha, and epsilon
#     """
#     # Set up Stiefel manifolds of the right dimensions. By convention d
#     St_Nk = Stiefel(N, k)
#     St_nk = Stiefel(n, k)
#     St_Nn = Stiefel(N, n)

#     # If no initial point given, generate one
#     if alpha is None:
#         alpha = St_Nn.random_point()

#     # Sample points in V_k(Rd) and map them into V_k(Rn)
#     xs = get_samples(St_nk, m)
#     initial_ys = [alpha.dot(x) for x in xs]

#     # Add noise to initial_ys
#     noise_vectors = [np.random.randn(N, k) for i in range(m)]
#     normalized_noise = [v/np.linalg.norm(v)*epsilon for v in noise_vectors]
#     perturbed_ys = np.add(initial_ys, normalized_noise)

#     # Project each back down to Stiefel, keeping signs which may be dropped in QR decomp
#     abs_ys = [np.linalg.qr(y)[0] for y in perturbed_ys]
#     sing_vals = [np.linalg.qr(y)[1] for y in perturbed_ys]
#     signs = np.array([np.sign(val) for val in sing_vals])
#     ys = abs_ys * signs

#     # Create dictionary with Manopt output and initial data, e.g. y_is, choice of alpha, etc.
#     sample_result = {'points':ys, 'alpha':alpha, 'epsilon':epsilon, 'xs':xs}

#     return sample_result