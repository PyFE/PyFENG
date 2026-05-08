import numpy as np
import scipy.special as spsp


def GHQ(n_quad, loc=0.0, scale=1.0):
    z, w, w_sum = spsp.roots_hermitenorm(n_quad, mu=True)
    w /= w_sum  # 1/np.sqrt(2.0 * np.pi)
    z = scale * z + loc
    return z, w


class NdGHQ:
    """
    N-dimensional Gauss-Hermite quadratuares.

    """

    sizes, powers, dim = np.array([]), np.array([]), 0
    z_list, w_list = [], []

    def __init__(self, sizes):
        self.sizes = np.array(sizes)
        powers = np.cumprod(self.sizes[::-1])[::-1]
        self.powers = np.append(powers, 1)
        self.total = self.powers[0]
        self.dim = len(self.sizes)
        self.z_list, self.w_list = [], []

        for size in sizes:
            z, w = spsp.roots_hermitenorm(size)
            w = w / np.sqrt(2.0*np.pi)
            self.z_list.append(z)
            self.w_list.append(w)

    def index_n(self, ind):
        ind_n = np.floor_divide(np.remainder(ind, self.powers[:-1]), self.powers[1:])
        return ind_n

    def indeces(self):
        ind_all = np.arange(self.total)
        return self.index_n(ind_all[:, None])

    def z_vec_weight(self, ind=None):

        if ind is None:  # return all
            z_vec = np.array([self.z_list[-1]])
            weight = self.w_list[-1]

            for k in range(self.dim - 2, -1, -1):
                z1 = np.repeat(self.z_list[k], z_vec.shape[1])
                z2 = np.tile(z_vec, len(self.z_list[k]))
                z_vec = np.vstack((z1, z2))

                w1 = np.repeat(self.w_list[k], len(weight))
                w2 = np.tile(weight, len(self.w_list[k]))
                weight = w1 * w2

            z_vec = z_vec.T

        else:  # return items for index
            ind_n = self.index_n(ind)
            z_vec = np.array(list(map(lambda ind_k: self.z_list[ind_k[0]][ind_k[1]], zip(range(self.dim), ind_n))))
            weight = np.prod(
                np.array(list(map(lambda ind_k: self.w_list[ind_k[0]][ind_k[1]], zip(range(self.dim), ind_n)))))

        return z_vec, weight
