"""
Szwagier, Tom, and Xavier Pennec.
"Nested subspace learning with flags." arXiv preprint arXiv:2502.06022 (2025).
"""

import numpy as np
from jax.scipy.linalg import expm
from pymanopt.manifolds.manifold import Manifold

class _FlagBase(Manifold):
    @property
    def typical_dist(self):
        raise NotImplementedError()

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def transport(self, point_a, point_b, tangent_vector_a):
        raise NotImplementedError()

    def zero_vector(self, point):
        raise NotImplementedError()

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self.projection(point, euclidean_gradient)

    def to_tangent_space(self, point, vector):
        return self.projection(point, vector)

class Flag(_FlagBase):

    def __init__(self, p: int, signature: tuple):
        self._p = p     # ambient dimension
        self._q = signature[-1]
        self._signature = signature
        self._signature_full = (0,) + signature + (p,)  # tuple concatenation
        self._type = np.diff(self._signature_full)

        if not ((0 < signature[0]) and (len(signature) == 1 or np.all(signature[:-1] < signature[1:])) and (
                self._q < p)):
            raise ValueError(
                f"Need 0 < signature[0] < ... < signature[-1] < p. Values supplied were p = {p} and signature = {signature}"
            )
        self.name = f"Fl({p},{signature})"
        self.dimension = int(p * (p - 1) / 2 - np.sum(self._type * (self._type - 1) / 2))
        super().__init__(self.name, self.dimension)

    def __repr__(self):
        return self.name

    def random_point(self):
        u, _, vt = np.linalg.svd(np.random.normal(size=(self._p, self._q)), full_matrices=False)
        return u @ vt

    def orthogonal_extension(self, point):
        """ get the orthogonal complement of the point and concatenate """
        point_perp = np.linalg.svd(point, full_matrices=True, compute_uv=True)[0][:, self._q:]

        return np.concatenate([point, point_perp], axis=1)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        G = np.eye(self._p) - 1 / 2 * point @ point.T
        return np.trace(tangent_vector_a.T @ G @ tangent_vector_b)

    def random_tangent_vector(self, point):
        """
        Corollary 4.12,
        Ye, Ke, Ken Sze-Wai Wong, and Lek-Heng Lim.
        "Optimization on flag manifolds." Mathematical Programming 194.1 (2022): 621-660.
        """

        point_completion = self.orthogonal_extension(point)
        B_upper = np.random.normal(size=(self._q, self._q))
        B_upper = (B_upper - B_upper.T) / 2
        B_upper = B_upper - np.diag(np.diag(B_upper))
        B_lower = np.random.normal(size=(self._p - self._q, self._q))
        B = np.concatenate([B_upper, B_lower], axis=0)

        return point_completion @ B

    def is_tangent_vector(self, direction):
        pass

    def projection(self, point, vector):
        Z = np.zeros_like(vector)
        for k in range(1, len(self._signature) + 1):
            U_k = point[:, self._signature_full[k-1]: self._signature_full[k]]
            X_k = vector[:, self._signature_full[k-1]: self._signature_full[k]]
            Z_k = (np.eye(self._p) - U_k @ U_k.T) @ X_k
            for l in range(1, len(self._signature) + 1):
                if l != k:
                    U_l = point[:, self._signature_full[l-1]: self._signature_full[l]]
                    X_l = vector[:, self._signature_full[l-1]: self._signature_full[l]]
                    Z_k -= U_l @ X_l.T @ U_k
            Z[:, self._signature_full[k-1]: self._signature_full[k]] = np.copy(Z_k)
        return Z

    def exponential_map(self, point, tangent_vector):
        """
        Proposition 4.14,
        Ye, Ke, Ken Sze-Wai Wong, and Lek-Heng Lim.
        "Optimization on flag manifolds." Mathematical Programming 194.1 (2022): 621-660.
        """

        point_completion = self.orthogonal_extension(point)
        B = np.zeros_like(point_completion)
        B[:, :self._q] = point_completion.T @ tangent_vector
        B[:, self._q:] = np.concatenate([-B[self._q:, :self._q].T,
                                          np.zeros((self._p - self._q, self._p - self._q))],
                                         axis=0)
        B = (B - B.T) / 2
        u, _, vt = np.linalg.svd(point_completion @ expm(B))
        return u @ vt[:, :self._q]  # for numerical stability, we project orthogonally


    def retraction(self, point, tangent_vector):
        u, _, vt = np.linalg.svd(point + tangent_vector, full_matrices=False)
        return u @ vt

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        """
        Proposition 6.1,
        Ye, Ke, Ken Sze-Wai Wong, and Lek-Heng Lim.
        "Optimization on flag manifolds." Mathematical Programming 194.1 (2022): 621-660.
        """
        partition = (0, ) + self._signature
        point_blocks = [point[:, partition[j]:partition[j + 1]] for j in range(len(partition) - 1)]
        euclidean_grad_blocks = [euclidean_gradient[:, partition[j]:partition[j + 1]] \
                                 for j in range(len(partition)-1)]
        riemannian_grad_blocks = []
        for i, (pt_block, euc_grad_block) in enumerate(zip(point_blocks, euclidean_grad_blocks)):
            comp1 = pt_block @ pt_block.T @ euc_grad_block
            comp2 = np.sum(point_blocks[j] @ euclidean_grad_blocks[j].T @ point_blocks[i] \
                            for j in range(len(partition)-1) if j!= i)
            riem_grad_block = euc_grad_block - (comp1 + comp2)
            riemannian_grad_blocks.append(riem_grad_block)

        riemnnian_gradient = np.hstack(riemannian_grad_blocks)

        return riemnnian_gradient