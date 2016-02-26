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
import itsne

# set up paths
output_path = 'outputs/mnist_imgs.html'

# load up features
data_dir = '../Data'
data_path = os.path.join(data_dir, 'mnist_train_raw_pixels.csv')

# load from data
df = pd.read_csv(data_path)
df['paths'] = df['uid'].apply(lambda x: os.path.join(data_dir,
                                                     'images/%i.png' % x))
img_paths = df.pop('paths')
uids = df.pop('uid')
labels = df.pop('label')

# load up images into giant array
imgs = [np.array(Image.open(img_path)) for img_path in img_paths]

# plot itsne in bokeh
xy = itsne.plot_tsne(output_path, df.values, uids=uids,
                     labels=labels, imgs=imgs, n_per_cluster=25,
                     img_alpha=190)
