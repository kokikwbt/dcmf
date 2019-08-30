#!/bin/env python3

""" 
"""

import numpy as np
import pandas as pd
from scipy.linalg import pinv
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

from contextual_matrix import get_contextual_matrix


class DCMF(object):
    """ DCMF: Dynamic Contextual Matrix Factorization
        for coevolving time series (In SDM 2015)

    Inputs
    ======
        - X: np.array
            multi-dimensional time series
            (with missing values)
        - rank: int
            the dimension of latent factor

    Outputs
    =======
        - 

    Math
    ====
        Eq.3.1:  z(t) = Bz(t-1) + gauss(0, sgmZ)
        Eq.3.2:  x(t) = W(t) * (Uz(t) + gauss(0, sgmX))
        Eq.3.3:  s(j) = Uv(j) + gauss(0, sgmV)

    """
    def __init__(self, X, S, rank, lmd=5e-1):
        self.__parse_inputs(X, rank, lmd)
        self.T = T = X.shape[0]
        self.N = N = X.shape[1]
        self.L = L = rank if rank < N else N
        # Definition 2.1: a network of time series (NoT)
        self.X = X
        self.W = ~np.isnan(X)
        self.S = S
        # hyper parameter
        self.lmd = lmd
        # model parameters:
        self.U = np.random.rand(N, L)
        self.B = np.random.rand(L, L)
        self.z0 = np.random.rand(L)
        self.psi0 = np.random.rand(L, L)
        self.sgmZ = np.random.rand()
        self.sgmX = np.random.rand()
        self.sgmS = np.random.rand()
        self.sgmV = np.random.rand()

    def __parse_inputs(self, X, rank, lmd):
        if X.ndim > 2:
            raise TypeError("input must be 1d/2d-array")
        if not type(rank) == int:
            raise TypeError("lmd must be an integer")
        if not 0 <= lmd <= 1:
            raise ValueError(" ")

    def em(self, max_iter=100, tol=1.e-7, verbose=False):
        """
        """

        for iteration in trange(max_iter, desc='EM'):
            """ E-step """
            zt, ztt, zt1t = self.backward(*self.forward())
            v, vv = self.compute_latent_context()

            """ M-step """
            self.z0 = zt[0]
            self.psi0 = ztt[0] - np.outer(zt[0], zt[0])
            self.B = self.update_transition_matrix(ztt, zt1t)
            self.sgmZ = self.update_transition_covariance(ztt, zt1t)
            self.sgmV = self.update_contextual_covariance(vv)

            for i in range(self.N):
                A1 = self.lmd / self.sgmS * sum([self.S[i, j] * v[j] for j in range(self.N)])
                A1 += (1 - self.lmd) / self.sgmX * sum([self.W[t, i] * self.X[t, i] * zt[t] for t in range(self.T)])
                A2 = self.lmd / self.sgmS * sum(vv)
                A2 += (1 - self.lmd) / self.sgmX * sum([self.W[t, i] * ztt[t] for t in range(self.T)])
                self.U[i, :] = A1 @ pinv(A2)

            self.sgmS = self.update_network_covariance(v, vv)
            self.sgmX = self.update_observation_covariance(zt, ztt)

            if verbose == True:
                pass

        # latent factors
        self.Z = zt
        self.V = np.vstack(v)

        return self.Z, self.U, self.V

    def forward(self):
        """ Perform the forward algorithm

        Note
        ====
            ot: the indices of the observed entries of xt
            Ht: the corresponding compressed version of U
            x(t) = H(t) @ z(t) + Gaussian noise
        """
        mu_ = np.zeros((self.T, self.L))
        psi = np.zeros((self.T, self.L, self.L))
        K = np.zeros((self.T, self.L, self.N))
        P = np.zeros((self.T, self.L, self.L))
        I = np.eye(self.L)

        for t in range(self.T):
            # Construct H(t) based on Eq. (3.6)
            ot = self.W[t, :]
            lt = sum(ot)
            It = np.eye(lt)
            x = self.X[t, ot]
            H = self.U[ot, :]

            # Estimate mu and phi: Eq. (3.8), (3.9)
            if t == 0:
                K[0] = self.psi0 @ H.T @ pinv(H @ self.psi0 @ H.T + self.sgmX * It)
                psi[0] = (I - K[0] @ H) @ self.psi0
                mu_[0] = self.z0 + K[0] @ (x - H @ self.z0)

            else:
                P[t-1] = self.B @ psi[t-1] @ self.B.T + self.sgmZ * I
                K[t] = P[t-1] @ H.T @ pinv(H @ P[t-1] @ H.T + self.sgmX * It)
                psi[t] = (I - K[t] @ H) @ P[t-1]
                mu_[t] = self.B @ mu_[t-1] + K[t] @ (x - H @ self.B @ mu_[t-1])

        return mu_, psi, K, P

    def backward(self, mu_, psi, K, P):
        """ Perform the backward algorithm
        """
        J = np.zeros((self.T, self.L, self.L))
        zt = np.zeros((self.T, self.L))
        ztt = np.zeros((self.T, self.L, self.L))
        zt1t = np.zeros((self.T, self.L, self.L))
        mu_h = np.zeros((self.T, self.L))
        psih = np.zeros((self.T, self.L, self.L))
        mu_h[-1], psih[-1] = mu_[-1], psi[-1]

        for t in reversed(range(self.T - 1)):
            J[t] = psi[t] @ self.B.T @ pinv(P[t])
            psih[t] = psi[t] + J[t] @ (psih[t+1] - P[t]) @ J[t].T
            mu_h[t] = mu_[t] + J[t] @ (mu_h[t+1] - self.B @ mu_[t])

        zt = mu_h

        for t in range(self.T):
            if t > 0: zt1t[t] = psih[t] @ J[t-1].T + np.outer(mu_h[t], mu_h[t-1])
            ztt[t] = psih[t] + np.outer(mu_h[t], mu_h[t])

        return zt, ztt, zt1t

    def compute_loglikelihood(self):
        """ Evaluate the objective function (Eq. 3.5)
        """
        llh = 0
        return llh

    def compute_latent_context(self):
        """ Eq. (3.12)
        """
        v = np.zeros((self.N, self.L))
        vv = np.zeros((self.N, self.L, self.L))
        Minv = pinv(self.U.T @ self.U + self.sgmS / self.sgmV * np.eye(self.L))
        gamma = self.sgmS * Minv

        for i in range(self.N):
            v[i] = Minv @ self.U.T @ self.S[i, :]
            vv[i] = gamma + np.outer(v[i], v[i])

        return v, vv

    def update_transition_matrix(self, ztt, zt1t):
        return sum(zt1t[1:]) @ pinv(sum(ztt[:-1]))

    def update_transition_covariance(self, ztt, zt1t):
        val = np.trace(
            sum(ztt[1:])
            - sum(zt1t[1:]) @ self.B.T
            - (sum(zt1t[1:]) @ self.B.T).T
            + self.B @ sum(ztt[:-1]) @ self.B.T
        )
        return val / ((self.T - 1) * self.L)

    def update_observation_covariance(self, zt, ztt):
        val = 0
        for t in range(self.T):
            ot = self.W[t, :]
            xt = self.X[t, ot]
            Ht = self.U[ot, :]
            val += np.trace(Ht @ ztt[t] @ Ht.T)
            val += xt @ xt - 2 * xt @ (Ht @ zt[t])

        return val / self.W.sum()

    def update_contextual_covariance(self, vv):
        return sum([np.trace(vv[i]) for i in range(self.N)]) / (self.N * self.L)

    def update_network_covariance(self, v, vv):
        val = sum([self.S[i].T @ self.S[i] - 2 * self.S[i] @ (self.U @ v[i]) for i in range(self.N)])
        val += np.trace(self.U @ sum(vv) @ self.U.T)
        return val / self.N ** 2

    def reconstruct(self):
        return (self.U @ self.Z.T).T

    def save(self, outdir, visualize=True):
        """ Save/Visualize results
        """
        np.savetxt(outdir+'Xorg.csv', self.X, delimiter=',')
        np.savetxt(outdir+'Xrec.csv', self.reconstruct(), delimiter=',')
        np.savetxt(outdir+'S.txt', self.S)
        np.savetxt(outdir+'U.txt', self.U)
        np.savetxt(outdir+'V.txt', self.V)
        np.savetxt(outdir+'Z.txt', self.Z)

        if visualize == True:
            plt.figure(figsize=(16, 8))
            plt.subplot(211)
            plt.plot(self.X)
            plt.title('Original sequence')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.subplot(212)
            plt.plot(self.reconstruct())
            plt.title(f'Reconstructed sequence (lmd= {self.lmd})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.savefig(outdir+'result.png')
            plt.close()

            sns.heatmap(self.S)
            plt.savefig(outdir+'S.png')
            plt.xlabel('Dimensions')
            plt.ylabel('Dimensions')
            plt.close()
            sns.heatmap(self.U)
            plt.savefig(outdir+'U.png')
            plt.xlabel('Latent dimensions')
            plt.ylabel('Dimensions')
            plt.close()
            sns.heatmap(self.V)
            plt.savefig(outdir+'V.png')
            plt.xlabel('Latent dimensions')
            plt.ylabel('Dimensions')
            plt.close()
            plt.plot(self.Z)
            plt.savefig(outdir+'Z.png')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.close()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    sns.set()

    outdir = './_out/example/'

    X = np.loadtxt('./_dat/86_11.amc.4d', delimiter=',')
    X = scale(X)
    N = X.shape[1]

    S = get_contextual_matrix(X, 0, method='cosine')

    rank = 3
    model = DCMF(X, S, rank, lmd=.1)
    model.em(max_iter=20)

    model.save(outdir)
