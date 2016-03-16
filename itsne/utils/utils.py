# COPYRIGHT
# ---------
# All contributions by Long Van Ho:
# Copyright (c) 2015 Long Van Ho
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015, the respective contributors.
# All rights reserved.
#
# LICENSE
# ---------
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN
# ==============================================================================

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
