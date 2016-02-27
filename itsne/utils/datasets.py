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
