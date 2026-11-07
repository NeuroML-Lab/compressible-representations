import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from numpy import ndarray


def compute_RSA(X: ndarray,Y: ndarray, dist_metric: str ='correlation', corr_metric: str ='pearson'):
    '''
    computes Representational Similarity between two representation matrices
    args:
        X: rep matrix 1 [n_stimuli x n_channels]
        Y: rep matrix 2 [n_stimuli x n_channels]
        dist_metric: metric used to establish dissimilarity and construct RDMs
            choices: 'correlation' (default), 'euclidean'
        corr_metric: metric used to establish similarity between RDMs
            choices: 'pearson' (default), 'spearman', 'kendall'

    output:
        rsa_corr: the final representational similarity score betwen RDMs of X&Y
    '''

    if dist_metric=='correlation':
        X_dist=1-np.corrcoef(X)
        Y_dist=1-np.corrcoef(Y)
    elif dist_metric=='euclidean':
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        X_dist=squareform(pdist(X, metric='euclidean'))
        Y_dist=squareform(pdist(Y, metric='euclidean'))
    else:
        raise ValueError(f"Unsupported corr_metric {corr_metric}.")

    X_dist_flat = X_dist[np.triu_indices(X_dist.shape[0], k=1)]
    Y_dist_flat = Y_dist[np.triu_indices(Y_dist.shape[0], k=1)]
    
    valid_indices = ~np.isnan(X_dist_flat) & ~np.isnan(Y_dist_flat)
    X_dist_flat = X_dist_flat[valid_indices]
    Y_dist_flat = Y_dist_flat[valid_indices]

    if len(X_dist_flat) < 2:
        rsa_corr= np.nan
    else:
      if corr_metric=='pearson':
        rsa_corr = pearsonr(X_dist_flat, Y_dist_flat)[0]
      elif corr_metric=='spearman':
        rsa_corr = spearmanr(X_dist_flat, Y_dist_flat)[0]
      elif corr_metric=='kendall':
        rsa_corr = kendalltau(X_dist_flat, Y_dist_flat)[0]
      else:
         raise ValueError(f"Unsupported corr_metric {corr_metric}.")

    
    return rsa_corr



def compute_CKA(X: ndarray, Y: ndarray, kernel: str ='linear'):
    '''
    Computes Centered Kernel Alignment (CKA) between two representation matrices.
    Implements CKA as in Kornblith et al. (2019) - "Similarity of Neural 
    Network Representations Revisited".

    args:
        X: rep matrix 1 [n_stimuli x n_channels]
        Y: rep matrix 2 [n_stimuli x n_channels]
        kernel: kernel type for computing Gram matrices
            choices: 'linear' (default)
                     'rbf' (RBF/Gaussian kernel, experimental)

    output:
        cka_score: CKA similarity score in [0, 1], where 1 = identical representations
    '''

    def centering(K):
        # Apply centering matrix H = I - (1/n)11^T
        n = K.shape[0]
        unit = np.ones((n, n))
        H = np.eye(n) - unit / n
        return H @ K @ H

    def hsic(K, L):
        # Unbiased HSIC estimator (Hilbert-Schmidt Independence Criterion)
        return np.sum(centering(K) * centering(L)) / ((K.shape[0] - 1) ** 2)

    def linear_kernel(Z):
        return Z @ Z.T

    def rbf_kernel(Z, sigma=None):
        dist_sq = squareform(pdist(Z, metric='sqeuclidean'))
        if sigma is None:
            sigma = np.median(dist_sq[dist_sq > 0])  # median heuristic
        return np.exp(-dist_sq / (2 * sigma))

    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same n_obs. Got {X.shape[0]} and {Y.shape[0]}.")
    if X.shape[0] < 4:
        return np.nan

    # Center columns (mean-subtract per channel) 
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    if kernel == 'linear':
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == 'rbf':
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"Unsupported kernel '{kernel}'")

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return np.nan

    cka_score = hsic_kl / denom
    return cka_score