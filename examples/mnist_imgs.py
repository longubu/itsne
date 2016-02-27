"""
Uses T-SNE to reduce feature dimensions to 2-dimensions for visualizations
from subset of MNIST dataset, using raw-pixel values for features.

Plots each datapoints using their corresponding images
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

# itsne imports
from itsne import itsne

# set up paths
output_path = 'outputs/mnist_imgs.html'

# load up features
data_dir = '../data'
data_path = os.path.join(data_dir, 'mnist_train_raw_pixels.csv')

# load from data
df = pd.read_csv(data_path)
uids = df.pop('uid')
labels = df.pop('label')

# check if directory of images exists
img_dir = os.path.join(data_dir, 'mnist_images')
if not os.path.exists(img_dir):
    print("Image database needed does not exists: %s" % img_dir)
    print("... Generating image database of mnist to %s" % img_dir)
    from itsne.utils import datasets
    datasets.generate_mnist_images(df.values, uids, img_dir)

# load up images into giant array, type(uids) == pd.Series so we can use apply
img_paths = uids.apply(lambda x: os.path.join(img_dir, '%i.png' % x))
imgs = [np.array(Image.open(img_path)) for img_path in img_paths]

# plot itsne in bokeh
xy = itsne.plot_tsne(output_path, df.values, uids=uids,
                     labels=labels, imgs=imgs, n_per_cluster=25,
                     img_alpha=190)
