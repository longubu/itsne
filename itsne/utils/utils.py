"""
General python utility functions
"""
import numpy as np


def update_dict(cfg, c):
    """Updates `cfg` dictionary in-place with another dictionary `c`, following
    its key-value (including nested dictionaries)"""
    for k, v in c.items():
        if k not in cfg:
            cfg[k] = {}

        if isinstance(v, dict):
            update_dict(cfg[k], v)
        else:
            cfg[k] = v


def get_rng_samples(labels, with_label, n_samples):
    """Get `n_samples` random indexes from labels such that
    labels == with_label. Returns ndarray of indexes"""
    labels = np.array(labels)  # make sure its ndarray for proper broadcasting
    weights = np.zeros(len(labels))
    weights[labels == with_label] = 1
    p = (weights / float(np.sum(weights)))
    idxs = np.random.choice(np.arange(len(labels)), size=n_samples,
                            replace=False, p=p).astype(int)

    return idxs


def sample_n_per_label(labels, n_samples):
    """Get `n_samples` random indexes from labels for each unique label in
    `labels`. Returns ndarray of indexes"""
    labels = np.array(labels)  # make sure its ndarray just in case
    unique_labels = np.unique(labels)
    idxs = []
    for u_lbl in unique_labels:
        idxs = np.concatenate(
                  [idxs, get_rng_samples(labels, u_lbl, n_samples)])

    return np.array(idxs, dtype=int)
