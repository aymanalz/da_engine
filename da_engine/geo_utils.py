import os, sys
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

def normal_to_mixGuassian(ens, mixing_rate = [], mu = [], sig = [], seed = None  ):
    """
    Convert an ensemble of Guassian realizations to mixed Guassian
    ens is (n by N) : an ensemble of random fields.
    n               : number of nodes
    N               : number of realizations
    mu              : a list or an array that contains means or modes
    sig             : a list or array that contains std for each mode
    mixing_rate     : the proportion of each distribution
    """

    n, N = ens.shape
    number_of_dists = len(mixing_rate)
    number_of_samples = 100000

    sum_mix = np.sum(mixing_rate)
    mixing_rate = [v/sum_mix for v in mixing_rate]

    if not(seed is None):
        np.random.seed(seed)

    for i in range(number_of_dists):
        nn = np.round(mixing_rate[i] * number_of_samples)
        val = mu[i] + sig[i] * np.random.randn(int(nn), 1)
        if i== 0:
            samples = val
        else:
            samples = np.vstack((samples, val))

    ecdf = ECDF(samples.flatten()) # target cdf
    ecdf1 = ECDF(ens.flatten())  # currnet cdf


    f = np.interp(ens, ecdf1.x, ecdf1.y)
    mask_inf = np.logical_not(np.isinf(ecdf.x))
    yy = np.interp(f, ecdf.y[mask_inf], ecdf.x[mask_inf])

    return yy
    

if __name__ == "__main__":
	ens = np.random.randn(1000, 1000)
	ens = normal_to_mixGuassian(ens, mixing_rate = [0.33, 0.33, 0.33], mu = [0.5, 1, 1.5], sig = [0.1,0.1,0.2], seed = None  )
	import matplotlib.pyplot as plt
	plt.hist(np.sort(yy.flatten()), bins=100)
	xx = 1







