import os
import shutil
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity


def get_contextual_matrix(X, alpha, method='standard'):

    if method == 'standard':
        return compute_network(X, alpha)

    elif method == 'cosine':
        return compute_cosine_sim(X)


def compute_network(X, alpha):
    raise NotImplementedError


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
    
    
def scale(data):
    return preprocessing.scale(data)


def minmax_scale(data):
    return preprocessing.minmax_scale(data)


def make_directory(path, inplace=True):
    if inplace == True:
        if os.path.exists(path):
            shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)


def plot_sequence(X, title, fn):
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(X)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        fig.savefig(fn, bbox_inches='tight')
        return fig, ax


def plot_component(A, title, fn):
    with sns.axes_style('white'):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(A, ax=ax)
        ax.set_title(title)
        fig.savefig(fn, bbox_inches='tight')
        return fig, ax


def plot_dcmf(outdir, object):

    plot_sequence(object.Z, 'Latent sequence: Z', outdir+'/Z.png')

    plot_component(object.B, 'Transition Matrix: B', outdir+'/B.png')
    plot_component(object.U, 'Object Latent Matrix: U', outdir+'/U.png')
    plot_component(object.V, 'Contextual Latent Matrix: V', outdir+'/V.png')
    plot_component(object.S, 'Contextual Matrix: S', outdir+'/S.png')
    plot_component(object.Y, 'Reconstructed Contextual Matrix: Y', outdir+'/Y.png')
