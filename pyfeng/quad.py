import numpy as np
import scipy.special as spsp


def GHQ(n_quad, loc=0.0, scale=1.0):
    z, w = spsp.roots_hermitenorm(n_quad)
    z = scale * z + loc
    w *= 0.3989422804014326779399  # 1/np.sqrt(2.0 * np.pi)
    return z, w


def Gamma(n_quad, shape=1.0, rate=1.0, scale=None):
    """"
    Quadrature for gamma distribution (shape, rate) from generalized Laguerre quadrataure.

    The PDF of the distribution is
        rate^shape / Gamma(shape) x^(shape-1) e^{-rate x}
    or
        1 / Gamma(shape) / scale^shape x^(shape-1) e^{-x/scale}

    Args:
        n_quad:
        shape:
        rate:
        scale:

    Returns:
        (points, weights)

    Examples:
        >>> import scipy.special as spsp
        >>> alpha, beta = 2, 2
        >>> x, w = Gamma(n_quad=9, shape=alpha, rate=beta)
        >>> sum(x*w), alpha/beta # mean of Gamma
        >>> sum(w/x), beta/(alpha-1) # mean of Igamma
        >>> # E( 1/sqrt(X) ) = 1/sqrt(theta) * Gamma(k-0.5) / Gamma(k)
        >>> sum( w / np.sqrt(x) ), np.sqrt(beta)*spsp.gamma(alpha-0.5)/spsp.gamma(alpha)
        (0.43816259125341767, 0.43827054723084935)
    """

    assert(shape > 0)
    scale = scale or 1/rate
    x, w = spsp.roots_genlaguerre(n_quad, shape - 1)

    x *= scale
    w /= w.sum()

    return x, w


def InvGauss(n_quad, mu=1.0, lam=1.0):
    """
    Quadrature for the inverse Gaussian (IG) disribution from Choi et al (2021).

    sqrt(lambda / 2pi x^3) exp(-lambda(x-mu)^2 / 2mu^2 x)

    Args:
        n_quad:
        mu:
        lam:

    Returns:
        (points, weights)

    References:
        - Choi J, Du Y, Song Q (2021) Inverse Gaussian quadrature and finite normal-mixture approximation of the generalized hyperbolic distribution. Journal of Computational and Applied Mathematics 388:113302. https://doi.org/10.1016/j.cam.2020.113302

    Examples:
        >>> mu, lam = 2, 1.5
        >>> x, w = InvGauss(n_quad=9, mu=mu, lam=lam)
        >>> sum(x*w), mu  # mean of IG
        >>> sum(w/x), (1/mu + 1/lam) )  # mean of 1/IG
        >>> sum((x - mu)**2*w), mu**3/lam  # variance of IG
        >>> sum((1/x - 1/mu - 1/lam)**2*w), (1/mu/lam + 2/lam**2)  # variance of 1/IG
    """
    z, w = spsp.roots_hermitenorm(n_quad)

    fac = 0.5 * mu / lam
    y_hat = np.square(z) * fac

    x = 1.0 + y_hat + z*np.sqrt(fac * (2.0 + y_hat))
    w *= 0.7978845608028653558799 / (1.0 + x)  ## np.sqrt(2.0/np.pi), 2*w from GHQ
    x *= mu

    return x, w


def GIG(n_quad, gamma=1, delta=1, p=-0.5, correct=False):
    """
    Quadrature for the generalized inverse Gaussian distribution (GIG) from Choi et al (2021).

    Args:
        n_quad:
        gamma:
        delta:
        p: -0.5 by default
        correct: normalize weights if True

    Returns:
        (points, weights)

    References:
        - Choi J, Du Y, Song Q (2021) Inverse Gaussian quadrature and finite normal-mixture approximation of the generalized hyperbolic distribution. Journal of Computational and Applied Mathematics 388:113302. https://doi.org/10.1016/j.cam.2020.113302

    Examples:
        >>> import scipy.special as spsp
        >>> gamma, delta, p = 1, 1, 0.2
        >>> x, w = GIG(n_quad=8, gamma=gamma, delta=delta, p=p, correct=False)
        >>> r = 0.3  # r-th moment
        >>> mom_r = np.power(delta/gamma, r) * spsp.kv(p+r, gamma*delta)/spsp.kv(p, gamma*delta)
        >>> mom_r, np.sum(np.power(x, r)*w)
    """

    mu = delta / gamma
    lam = delta * delta

    x, w = InvGauss(n_quad=n_quad, mu=mu, lam=lam)
    w *= np.power(x, p+0.5)

    if correct:
        w /= sum(w)
    else:
        ratio = np.power(gamma/delta, p)*np.exp(-gamma*delta)/delta/np.sqrt(2/np.pi)/spsp.kv(p, gamma*delta)
        w *= ratio

    return x, w


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
