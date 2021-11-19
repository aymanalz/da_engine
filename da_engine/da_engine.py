__author__ = "Ayman Alzraiee"
__version__ = "1.0.1"
__email__ = "aalzraiee@usgs.gov"

import numpy as np
from scipy.linalg import blas as blas
from scipy.linalg import lapack as lap
from scipy.stats import ortho_group


class Analysis(object):
    def __init__(self, K=None, H=None, D=None, d=None, R=None,
                 method='enkf', err_std=None, err_perc=None, E=None,
                 truncation=None, truncation_percent=0.01, verbose=True):

        """
        Data Assimilation tools
        Computes the analysed ensemble for K using the EnKF or square root schemes.

        Parameters
        ----------

        n             : number of parameters/states to be updated
        m             :number of measurements
        N             :Ensemble size
        H(m, N)       : Ensemble of model predictions at locations and times of observation
        K(n, N)       : Ensemble of of parameters/states to be updated
        d(m)          : observations
        mode          : Two methods are supported [EnKF, SRKF]
        truncation    : fraction of eigen vector to be removed


        """

        if verbose:
            print(" Analysis: verbose is on")

        # Get and check the dimensions
        m1, N1 = H.shape
        n1, N2 = K.shape

        if not (E is None):
            m2, N3 = E.shape
            if m1 != m2:
                ValueError("    The number of measurements (m) must be the same in H and E")
            if not (N1 == N2):
                ValueError("    The number of realizations (n) must be the same in H and E")

        elif not (R is None):
            m3, m3 = E.shape
            if m1 != m3:
                ValueError("    The number of measurements (m) must be the same in H and R")
        else:
            if verbose:
                print("    E and R are not defined. Meas. Error will be infered or computed")

            if d is None:
                print("    Measurements are not supplied. Try to compute measurement error from perturbed "
                      "measurements")
                m4, N4 = D.shape
                if not (m4 == m1):
                    ValueError("    The number of measurements (m) must be the same in H and D")
                if D is None:
                    ValueError("    You need to supply d, D, E, or R to compute measurement errors")
                if verbose:
                    print("    Compute measurement error from D by assuming the average of D is d")
                d = np.mean(D, axis=1)
                E = D - d.reshape(m1, 1)
                E - E.mean(axis=1).reshape(m1, 1)


            else:
                if 1 in d.shape or d.ndim == 1:
                   d = d.reshape(m1, 1)

                if D is None:
                    print("    Trying to generate meas. errors form user defined errors value or percentage")
                    if not (err_std is None):
                        E = err_std * np.random.randn(m1, N1)
                    elif not (err_perc is None):
                        err_std = err_perc * np.std(H, axis=1)
                        E = err_std * np.random.randn(m1, N1)
                    else:
                        print(" No information is available to compute measurement errors")
                else:
                    E = D - d
                    E - E.mean(axis=1).reshape(m1, 1)

        if not (method.lower() in ['enkf', 'sqrtkf']):
            ValueError(" Mode is supported")

        # generate D
        if D is None:
            if d is None:
                ValueError("Error : Obs vector is not passed.")
            if E is None:
                ValueError("Error: measurements can not be perturbed")
            if 1 in d.shape:
                while np.rank(d) > 1:
                    d = np.squeeze(d)
            d = d.reshape(len(d), 1)

            D = d + E

        if verbose: print('      analysis: Ensemble Kalman Filter (EnKF)')
        self.K = K
        self.H = H
        self.R = R
        self.E = E
        self.D = D
        self.d = d
        self.truncation = truncation
        self.truncation_percent = truncation_percent
        self.verbose = verbose
        self.method = method.lower()

    def update(self):
        if self.method == 'enkf':
            Ka = self.EnKF()
        elif self.method == 'sqrtkf':
            Ka = self.Sqrt_KF()
        else:
            raise ValueError("Unknown method...")

        return Ka

    def EnKF(self):

        """

        Returns
        -------

        """

        # Compute ensemble prior means
        inflation_factor = 1.0

        H_dash = self.H - np.mean(self.H, axis=1)[:, np.newaxis]
        HE = H_dash + inflation_factor * self.E
        K_dash = self.K - np.mean(self.K, axis=1)[:, np.newaxis]
        D_dash = self.D - self.H

        # compute inverse of C = H'H' + R

        # SVD of matrix C
        u, s, vt, ierr = lap.dgesvd(HE)
        if ierr != 0: ValueError('Sqrt_KF: ierr from call dgesvd = {}'.format(ierr))

        s_ = np.power(s, 2.0)
        sums_ = np.sum(s_)
        if self.truncation is None:
            s_perc = s_ / np.sum(s_)
            truncation = self.truncation_percent / 100.0
            s_ = s_[s_perc >= truncation]
        else:
            truncation = self.truncation
            s_ = s_[s_ >= truncation]

        p = len(s_)
        print('      analysis: dominant sing. values and'
              ' share {}, {}'.format(p, 100.0 * (np.sum(s_) / sums_)))

        s_ = 1.0 / s_

        u = u[:, 0:p]

        x1 = s_[:, np.newaxis] * u.T
        x2 = blas.dgemm(alpha=1, a=x1, b=D_dash)
        x3 = blas.dgemm(alpha=1, a=u, b=x2)
        x4 = blas.dgemm(alpha=1, a=H_dash.T, b=x3)

        Aa = self.K + blas.dgemm(alpha=1, a=K_dash, b=x4)

        return Aa

    def Sqrt_KF(self):

        """
        Compute mean least square and ensemble perturbations.

        """

        # Compute ensemble prior means
        prior_k_mean = np.mean(self.K, axis=1)
        prior_h_mean = np.mean(self.H, axis=1)
        innov = self.d - prior_h_mean
        n, N = self.K.shape

        H_dash = self.H - np.mean(self.H, axis=1)[:, np.newaxis]
        HE = H_dash + self.E
        K_dash = self.K - np.mean(self.K, axis=1)[:, np.newaxis]

        # SVD of matrix C
        u, s, vt, ierr = lap.dgesvd(HE)
        if ierr != 0: ValueError('Sqrt_KF: ierr from call dgesvd = {}'.format(ierr))

        sig = s.copy()

        s_ = np.power(s, 2.0)
        sums_ = np.sum(s_)
        if self.truncation is None:
            s_perc = s_ / np.sum(s_)
            truncation = self.truncation_percent / 100.0
            s_ = s_[s_perc >= truncation]
        else:
            truncation = self.truncation
            s_ = s_[s_ >= truncation]

        p = len(s_)
        print('      analysis: dominant sing. values and'
              ' share {}, {}'.format(p, 100.0 * (np.sum(s_) / sums_)))

        s_ = 1.0 / s_

        u_ = u[:, 0:p]

        x1 = s_[:, np.newaxis] * u_.T
        x2 = blas.dgemv(alpha=1, a=x1, x=innov)
        x3 = blas.dgemv(alpha=1, a=u_, x=x2)
        x4 = blas.dgemv(alpha=1, a=H_dash.T, x=x3)

        Ka = prior_k_mean + blas.dgemv(alpha=1, a=K_dash, x=x4)

        # compute C^-1
        sig = np.power(sig[0:p], -2.0)
        x2 = sig[:, np.newaxis] * u_.T
        c_1 = blas.dgemm(alpha=1, a=u_, b=x2)

        # I - Y(C^-1)Y
        c_1 = blas.dgemm(alpha=1, a=c_1, b=H_dash)
        c_1 = blas.dgemm(alpha=1, a=H_dash.T, b=c_1)
        diag = 1 - np.diag(c_1)
        np.fill_diagonal(c_1, diag)

        # decompose I - Y(C^-1)Y
        u2, sig2, vt2, ierr = lap.dgesvd(c_1)
        sig2[sig2 < 0] = 0
        sig2 = np.power(sig2, 0.5)
        p2 = len(sig2)
        if p2 < N:
            sig2 = np.append(sig2, np.zeros(N - p2))
        x2 = u2 * sig2[np.newaxis, :]
        x2 = blas.dgemm(alpha=1.0, a=x2, b=vt2)

        x2 = blas.dgemm(alpha=1, a=K_dash, b=x2)
        theta = ortho_group.rvs(N)
        x2 = blas.dgemm(alpha=1, a=x2, b=theta.T)
        Aa = Ka[:, np.newaxis] + x2

        return Aa
