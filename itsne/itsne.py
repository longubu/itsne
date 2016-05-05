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
Interactive visualizations of data using t-SNE clustering with Bokeh
"""
import bokeh.plotting as bkp
import bokeh.models as bkm
import numpy as np
from tsne import bh_sne
from sklearn.cluster import KMeans

# local imports
from utils import colors, bokeh_utils, utils


def plot_tsne(output_path, X, uids=None, labels=None, imgs=None,
              n_clusters=10, n_per_cluster=None, img_alpha=255):
    """Plots interactive visualization of data using t-SNE clustering
    and bokeh.

    Parameters
    -------
    output_path: str
        Path to save bokeh interaction. Should be extension .html

    X: array, shape (n_samples, n_features) or (n_samples, n_samples)
        Feature array to apply t-SNE clustering.

    uids: array, shape (n_samples,), default=None
        Uids associated with rows of X to use as tooltip for hover.
        If None, will not use for tooltip or anything.

    labels: array, shape (n_samples,), default=None
        Class labels or array of integers grouping each row of X
        to 1) color glyphs on scatter plot and 2) for tooltip of hover
        If None, will automatically compute clusters using KMeans
        with `n_clusters`

    imgs: array, shape (n_samples, _img_shape_)
       Images to be plotted with coordinates created from tsne.
       If None, will use circles (scatter-plot) of bokeh per data.

    n_clusters: int, default=10
        Number of clusters to compute using KMeans, if labels is not
        provided.

    n_per_cluster: int, default=None
        Number of items to plot per cluster. If None, will plot all items

    img_alpha: int, default=255
        If `imgs` is provided, will plot imgs with a transparency given by
        img_alpha. `img_alpha` must be in the range [0, 255], where 0 is
        completely transparent, and 255 is opaque.

    Returns
    -------
    xy, ndarray of shape (n_samples, 2)
        x & y coordinates output from t-sne collapsing of X features.

    Raises
    -------
    RuntimeError:
        Error checking for correct shapes of X, uids, labels, imgs, etc.

    Examples
    -------
    See `itsne/examples` for usage.
    """
    # fit tsne coordinates
    xy = bh_sne(X, pca_d=None, perplexity=30., theta=0.5)

    # get H, W from image by loading first image of the set, if provided
    if imgs is not None:
        imgs = np.array(imgs)  # make sure its numpy array for ease use
        # check if imgs correspond to X by size
        if len(imgs) != len(X):
            raise RuntimeError("len(imgs) != len(X) (%s != %s)"
                               % (len(imgs), len(X)))

        # get image shapes to convert to img object for bokeh to absorb
        shape = imgs[0].shape
        # greyscale
        if len(shape) == 2:
            H, W = shape
        elif len(shape) == 3:
            H, W, C = shape
        else:
            raise RuntimeError("Can't get correct image shape from first"
                               " image of the dataset. Got shape = %s" % shape)
        bh, bw = (H / 2, W / 2)
        scale = np.max([H, W]) / 5.0
        xy = xy * scale

        # plot glyphs for hover but not to show
        glyph_kwargs = {'alpha': 0.0, 'size': int(np.min([H, W]))}
    else:
        # no images provided, so lets show circle glyphs with hover
        glyph_kwargs = {'fill_alpha': 0.35, 'line_alpha': 0.9,
                        'line_width': 2, 'size': 12}

    # if no label is provided, color by KMeans clustering algorithm
    if labels is None:
        lbls = KMeans(n_clusters=n_clusters).fit_predict(X)
    else:
        if len(labels) != len(X):
            raise RuntimeError("len(labels) != len(X) (%s != %s)"
                               % (len(labels), len(X)))
        lbls = labels

    # To not overpopulate plot, plot only n data points per cluster label
    if n_per_cluster is None:
        idxs = np.arange(len(lbls))
    else:
        # get n_per_cluster per unique label in lbls.
        idxs = utils.sample_n_per_label(lbls, n_per_cluster)

    # get xy coordinates and prepare data for bokeh
    x = xy[idxs, 0]
    y = xy[idxs, 1]

    data = dict(x=x, y=y)
    hover_tt = []  # hover tool
    # fill dict and hover tool tip with uids & labels
    if uids is not None:
        if len(uids) != len(X):
            raise RuntimeError("len(uids) != len(X) (%s != %s)"
                               % (len(uids), len(X)))

        uids = uids[idxs]
        data['uids'] = uids
        hover_tt.append(('uid', '@uids'))

    if labels is not None:
        lbls = lbls[idxs]
        data['labels'] = lbls
        hover_tt.append(('label', '@labels'))

    # get color array per cluster
    c_arr = colors.get_color_arr(lbls, normed=False)
    hex_arr = colors.rgb_to_hex(c_arr)

    # get axis limits
    min_x, max_x = np.min(x), np.max(y)
    min_y, max_y = np.min(y), np.max(y)

    # finally, start plotting in bokeh
    bkp.output_file(output_path)
    p = bkp.figure(plot_width=1200, plot_height=800,
                   x_range=[min_x - np.abs(0.10 * min_x), max_x + .10 * max_x],
                   y_range=[min_y - np.abs(0.10 * min_y), max_x + .10 * max_y])

    source = bkp.ColumnDataSource(data=data)
    hover = bkm.HoverTool(tooltips=hover_tt)
    p.circle('x', 'y', source=source, fill_color=hex_arr, line_color=hex_arr,
             **glyph_kwargs)
    p.add_tools(hover)

    # if images provided, plot them on x&y coordinates instead of circle glyphs
    if imgs is not None:
        # Can't find any documentation on plotting an array of images for now..
        for i, img in enumerate(imgs[idxs]):
            bimg = bokeh_utils.preproc_img(img, alpha=img_alpha)
            p.image_rgba(image=[bimg], x=[x[i] - (bw / 2)],
                         y=[y[i] - (bh / 2)], dw=[bw], dh=[bh])

    # save bokeh plot
    bkp.save(p)
    return xy
