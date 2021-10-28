import argparse
import warnings
import time
import numpy as np

try:
    import utils
except:
    from . import utils


class DCMF:
    """ Dynamic Contextual Matrix Factorization
        for coevolving time series (SDM 2015)

        z_t|z_t-1 ~ Bz_t-1 + noise
        x_t|z_t ~ Uz_t + noise
        s_j|v_j ~ Uv_j + noise

        i.e., 

        X = UZ and S = UV (Thus, Joint Matrix Factorization)

    """
    def __init__(self, alpha=5e-1):
        self.alpha = alpha

    def initialize(self, X, rank, random_init=True):

        self.T = X.shape[0]
        self.N = X.shape[1]
        self.L = rank

        # Model parameters

        if random_init:
            self.U    = np.random.rand(self.N, self.L)
            self.B    = np.random.rand(self.L, self.L)
            self.z0   = np.random.rand(self.L)
            self.psi0 = np.random.rand(self.L, self.L)
            self.sgmZ = np.random.rand()
            self.sgmX = np.random.rand()
            self.sgmS = np.random.rand()
            self.sgmV = np.random.rand()
        else:
            self.U    = np.eye(self.N, self.L)
            self.B    = np.eye(self.L)
            self.z0   = np.zeros(self.L)
            self.psi0 = np.eye(self.L)
            self.sgmZ = self.sgmX = self.sgmS = self.sgmV = 1.

        # Workspace

        self.mu_ = np.zeros((self.T, self.L))
        self.psi = np.zeros((self.T, self.L, self.L))
        self.K   = np.zeros((self.T, self.L, self.N))
        self.P   = np.zeros((self.T, self.L, self.L))
        self.I   = np.eye(self.L)

        self.J    = np.zeros((self.T, self.L, self.L))
        self.zt   = np.zeros((self.T, self.L))
        self.ztt  = np.zeros((self.T, self.L, self.L))
        self.zt1t = np.zeros((self.T, self.L, self.L))
        self.mu_h = np.zeros((self.T, self.L))
        self.psih = np.zeros((self.T, self.L, self.L))

        self.v  = np.zeros((self.N, self.L))
        self.vv = np.zeros((self.N, self.L, self.L))

    def fit(self, X, S, rank, max_iter=20,
            random_init=True, tol=0.01, verbose=0):
        """ The EM algorithm for DCMF (Algorithm 1) """

        W = ~np.isnan(X)
        self.initialize(X, rank, random_init)
        history = []

        for iteration in range(max_iter):
            tic = time.process_time()

            """ E-step """
            llh = self.forward(X, W, return_loglikelihood=True)
            self.backward()
            self.update_latent_context(S)

            """ M-step """
            llh += self.solve_dcmf(X, W, S, return_loglikelihood=True)
            history.append(llh)

            toc = time.process_time()

            if verbose > 0:
                message = f'iter= {iteration+1}, '
                message += f'llh= {llh:.3f}, '
                message += f'time= {toc-tic:.3f} [sec]'
                print(message)

            if iteration > 2:
                rate = np.abs(history[-1] - history[-2]) / np.abs(history[-2])
                if rate < tol:
                    break

        else:
            message = "the EM algorithm did not converge\n"
            message += "Consider increasing 'max_iter'"
            warnings.warn(message)

        self.S = S
        self.Z = self.mu_h
        self.V = self.v
        self.Y = self.U @ self.V.T

        return history

    def forward(self, X, W, return_loglikelihood=False):

        llh = 0

        for t in range(self.T):

            ot = W[t]
            xt = X[t, ot]
            It = np.eye(xt.shape[0])
            Ht = self.U[ot, :]

            if t == 0:
                psi = self.psi0
                self.mu_[0] = self.z0

            else:
                self.P[t-1] = self.B @ self.psi[t-1] @ self.B.T + self.sgmZ * self.I
                psi = self.P[t-1]
                self.mu_[t] = self.B @ self.mu_[t-1]

            delta = xt - Ht @ self.mu_[t]
            sigma = Ht @ psi @ Ht.T + self.sgmX * It
            inv_sigma = np.linalg.pinv(sigma)

            self.K[t] = psi @ Ht.T @ inv_sigma
            self.mu_[t] += self.K[t] @ delta
            self.psi[t] = (self.I - self.K[t] @ Ht) @ psi

            if return_loglikelihood:
                df = delta @ inv_sigma @ delta / 2
                sign, logdet = np.linalg.slogdet(inv_sigma)
                llh -= self.L / 2 * np.log(2 * np.pi)
                llh += sign * logdet / 2 - df

        return llh

    def backward(self):

        self.mu_h[-1] = self.mu_[-1]
        self.psih[-1] = self.psi[-1]

        for t in reversed(range(self.T - 1)):
            self.J[t] = self.psi[t] @ self.B.T @ np.linalg.pinv(self.P[t])
            self.psih[t] = self.psi[t] + self.J[t] @ (self.psih[t+1] - self.P[t]) @ self.J[t].T
            self.mu_h[t] = self.mu_[t] + self.J[t] @ (self.mu_h[t+1] - self.B @ self.mu_[t])

        self.zt[:] = self.mu_h[:]

        for t in range(self.T):
            if t > 0:
                self.zt1t[t] = self.psih[t] @ self.J[t-1].T
                self.zt1t[t] += np.outer(self.mu_h[t], self.mu_h[t-1])

            self.ztt[t] = self.psih[t] + np.outer(self.mu_h[t], self.mu_h[t])

    def solve_dcmf(self, X, W, S, return_loglikelihood=False):
        """ Line 13 in Algorithm 1 """

        # Compute Equation (3.14)
        self.z0 = self.zt[0]
        self.psi0 = self.ztt[0] - np.outer(self.zt[0], self.zt[0])
        self.B = self.update_transition_matrix()
        self.sgmV = self.update_contextual_covariance()
        self.sgmZ = self.update_transition_covariance()

        # Compute Equation (3.15)
        llh = self.update_object_latent_matrix(X, W, S, return_loglikelihood)
        
        # Compute Equation (3.16)
        self.sgmS = self.update_network_covariance(S)
        self.sgmX = self.update_observation_covariance(X, W)

        return llh

    def update_latent_context(self, S):
        """ Equation (3.12)
        """
        Minv = np.linalg.pinv(self.U.T @ self.U + self.sgmS / self.sgmV * np.eye(self.L))
        gamma = self.sgmS * Minv

        for i in range(self.N):
            self.v[i] = Minv @ self.U.T @ S[i, :]
            self.vv[i] = gamma + np.outer(self.v[i], self.v[i])

    def update_transition_matrix(self):
        return sum(self.zt1t[1:]) @ np.linalg.pinv(sum(self.ztt[:-1]))

    def update_transition_covariance(self):

        val = np.trace(
            sum(self.ztt[1:])
            - sum(self.zt1t[1:]) @ self.B.T
            - (sum(self.zt1t[1:]) @ self.B.T).T
            + self.B @ sum(self.ztt[:-1]) @ self.B.T
        )

        return val / ((self.T - 1) * self.L)

    def update_object_latent_matrix(self, X, W, S, return_loglikelihood=False):
        """ Equation (3.15) """
        llh = 0

        for i in range(self.N):
            A1 = self.alpha / self.sgmS * sum(
                S[i, j] * self.v[j] for j in range(self.N))
            A1 += (1 - self.alpha) / self.sgmX * sum(
                W[t, i] * X[t, i] * self.zt[t] for t in range(self.T))
            A2 = self.alpha / self.sgmS * sum(self.vv)
            A2 += (1 - self.alpha) / self.sgmX * sum(
                W[t, i] * self.ztt[t] for t in range(self.T))
            self.U[i, :] = A1 @ np.linalg.pinv(A2)

        if return_loglikelihood:
            for i in range(self.N):
                # https://www.seas.ucla.edu/~vandenbe/publications/covsel1.pdf
                delta = S[i] - self.U[i] @ self.v[i]
                sigma = self.sgmS * np.eye(self.N) + self.U @ self.vv[i] @ self.U.T
                inv_sigma = np.linalg.pinv(sigma)
                sign, logdet = np.linalg.slogdet(inv_sigma)
                llh -= self.L / 2 * np.log(2 * np.pi)
                llh += sign * logdet / 2 - delta @ inv_sigma @ delta / 2

        return llh

    def update_observation_covariance(self, X, W):

        val = 0

        for t in range(self.T):
            ot = W[t, :]
            xt = X[t, ot]
            Ht = self.U[ot, :]
            val += np.trace(Ht @ self.ztt[t] @ Ht.T)
            val += xt @ xt - 2 * xt @ (Ht @ self.zt[t])

        return val / W.sum()

    def update_contextual_covariance(self):
        return sum(np.trace(self.vv[i]) for i in range(self.N)) / (self.N * self.L)

    def update_network_covariance(self, S):

        val = sum(S[i].T @ S[i] - 2 * S[i] @ (self.U @ self.v[i]) for i in range(self.N))
        val += np.trace(self.U @ sum(self.vv) @ self.U.T)
        return val / self.N ** 2

    def reconstruct(self):
        return (self.U @ self.Z.T).T

    def predict(self):
        """ Return smoothed data """
        return

    def forecast(self, forecast_step=1):
        """ Produce forecasts """
        return

    def fit_forecast(self, X, forecast_step=1):
        """ Estimate latent sequence and forecast future values """
        return

    def save(self, outdir, visualize=True):
        """ Save the DCMF parameters """

        np.savetxt(outdir+'/B.txt', self.B)
        np.savetxt(outdir+'/U.txt', self.U)
        np.savetxt(outdir+'/V.txt', self.V)
        np.savetxt(outdir+'/Y.txt', self.Y)
        np.savetxt(outdir+'/Z.txt', self.Z)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='out')
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--tol', type=float, default=0.01)
    parser.add_argument('--demo', action='store_true')
    config = parser.parse_args()

    utils.make_directory(config.output_dir)

    if config.demo:
        
        # Load time series
        X = np.loadtxt('data/86_11.amc.4d', delimiter=',')
        X = utils.scale(X)

        # Load contextual information
        S = utils.get_contextual_matrix(X, 0, method='cosine')

        model = DCMF(alpha=config.alpha)
        model.fit(X, S,
            rank=config.n_components,
            max_iter=config.max_iter,
            tol=config.tol,
            verbose=1)

        model.save(config.output_dir)
        utils.plot_dcmf(config.output_dir, model)

        exit()

