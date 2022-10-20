from sklearn.mixture import GaussianMixture as GMM
import numpy as np
def fisher_vector(xx, gmm):
    # xx = np.atleast_2d(xx)
    N = xx.shape[0]
    print(np.shape(xx))
    Q = gmm.predict_proba(xx)  # NxK
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))