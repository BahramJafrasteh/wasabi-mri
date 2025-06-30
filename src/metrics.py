import numpy as np
from scipy.stats import wasserstein_distance, energy_distance
from scipy.spatial.distance import jensenshannon
import ot
from collections import defaultdict
from scipy.spatial.distance import jensenshannon

def kl_divergence(data1, data2):
    mu1 = np.mean(data1, axis=0)
    mu2 = np.mean(data2, axis=0)
    cov1 = np.cov(data1, rowvar=False)
    cov2 = np.cov(data2, rowvar=False)
    k = mu1.shape[0]

    term1 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    term2 = np.trace(np.linalg.solve(cov2, cov1))
    term3 = (mu2 - mu1).T @ np.linalg.solve(cov2, mu2 - mu1)

    return 0.5 * (term1 + term2 + term3 - k)

def wasserstein_distance_pot(data1, data2):
    n_samples = 500
    M = ot.dist(data1, data2)
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    return ot.emd2(a, b, M)

def total_variation_distance(P, Q):
    return 0.5 * np.sum(np.abs(P - Q))

def hellinger_distance(P, Q):
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q))**2))

def bhattacharyya_distance(P, Q):
    return -np.log(np.sum(np.sqrt(P * Q)))




def compute_metrics(data_pos, data_neg=None, K=1000, criterion='wasabi'):
    """
    Compute traditional distance-based metrics between two datasets.

    Parameters
    ----------
    data_pos : np.ndarray
        Positive (typically real) volume data, shape [N, D].
    data_neg : np.ndarray or None
        Negative (typically synthetic) volume data, shape [M, D]. If None, will split `data_pos` randomly.
    K : int
        Number of bootstrap iterations.
    criterion : str
        Distance metric to use. Options: 'wasserstein', 'kl', 'jensenshannon',
        'tv', 'hellinger', 'bhattacharyya', 'energy'.

    Returns
    -------
    dict
        Dictionary with K computed distances, keyed by bootstrap iteration.
    """
    dict_metric = defaultdict(list)

    for k in range(K):
        print(f"Computing {criterion} - iteration {k+1}/{K}")

        # Sampling
        if data_neg is not None:
            # Sample up to 500 points from each
            idx_pos = np.random.choice(data_pos.shape[0], min(500, data_pos.shape[0]), replace=False)
            idx_neg = np.random.choice(data_neg.shape[0], min(500, data_neg.shape[0]), replace=False)
        else:
            # Sample 1000 from data_pos, split in half
            idx_all = np.random.choice(data_pos.shape[0], 1000, replace=False)
            idx_pos = idx_all[:500]
            idx_neg = idx_all[500:]
            data_neg = data_pos  # for symmetry

        pos_sample = data_pos[idx_pos]
        neg_sample = data_neg[idx_neg]

        # Compute chosen distance
        if criterion == 'wasabi':
            dist = wasserstein_distance_pot(pos_sample, neg_sample)
        elif criterion == 'kl':
            dist = kl_divergence(pos_sample, neg_sample)
        elif criterion == 'jensenshannon':
            dist = jensenshannon(pos_sample.mean(axis=0), neg_sample.mean(axis=0))
        elif criterion == 'tv':
            dist = total_variation_distance(pos_sample, neg_sample)
        elif criterion == 'hellinger':
            dist = hellinger_distance(pos_sample, neg_sample)
        elif criterion == 'bhattacharyya':
            dist = bhattacharyya_distance(pos_sample, neg_sample)
        elif criterion == 'energy':
            dist = energy_distance(pos_sample, neg_sample)
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

        dict_metric[k] = dist

    return dict_metric

