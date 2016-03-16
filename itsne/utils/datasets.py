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
Utility functions for generating necessary datasets for example visualizations.
"""
import numpy as np
import os
from PIL import Image


def generate_mnist_images(X, uids, output_dir):
    """Generates a directory to store mnist images in .png format.

    Parameters
    ------
    X: ndarray of shape (n_samples, 28 * 28)
        Array containing raw pixel values per mnist sample

    uids: ndarray of shape (n_samples,)
        Arrray corresponding to X with the associates unique ids.

    output_dir: str
        Directory to output images to
    """
    if len(X) != len(uids):
        raise RuntimeError("len(X) != len(uids) (%s != %s)"
                           % (len(X), len(uids)))

    # make dir if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    H, W = (28, 28)  # to reshape X to image size
    for uid, x in zip(uids, X):
        img = Image.fromarray(x.reshape(H, W).astype(np.uint8))
        img.save(os.path.join(output_dir, '%s.png' % uid))
