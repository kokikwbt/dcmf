#!/bin/env python3

from itertools import combinations_with_replacement
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_contextual_matrix(X, alpha, method='standard'):
    """
    """
    if method == 'standard':
        return compute_network(X, alpha)
    elif method == 'cosine':
        return compute_cosine_sim(X)


def compute_network(X, alpha):
    return S


def compute_cosine_sim(X):
    """ Define contextual matrix by the cosine similarity

    NOTE
    ====
        When your dataset is no network structure data,
        use the cosine similarity between each pair of
        time seires to define contextual matrix.

    """
    N = X.shape[1]
    S = np.zeros((N, N))

    for i, j in combinations_with_replacement(range(N), 2):
        Xi = X[:, i].reshape(1, -1)
        Xj = X[:, j].reshape(1, -1)
        S[i, j] = abs(cosine_similarity(Xi, Xj))

    S = S + S.T - np.diag(np.diag(S))

    return S