"""
Uses T-SNE to reduce feature dimensions to 2-dimensions for visualizations
from subset of MNIST dataset, using raw-pixel values for features.

Plots each datapoints as bokeh circles.
"""
import os
import pandas as pd

# itsne imports
import itsne

# set up paths
output_path = 'outputs/mnist_circles.html'

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

# plot itsne in bokeh
xy = itsne.plot_tsne(output_path, df.values, uids=uids,
                     labels=labels, imgs=None, n_per_cluster=None)
