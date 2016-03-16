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
Support functions for selecting colors.
"""
import numpy as np

# list taken from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau20 = np.array(tableau20)


def hex_to_rgb(value):
    """Converts hex string to RGB tuple of length 3"""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    """Converts iterable rgb array to hex string. Can also be an iterable
    containing rgb arrays, in which this will return a list of hex strings"""
    rgb = np.array(rgb)
    shape = rgb.shape
    # if tuple/iterable of len(3) -- aka single RGB
    if shape == (3L,):
        ret = '#%02x%02x%02x' % tuple(rgb)
    # else if rgb is a batch of rgbs, recursively use rgb_to_hex
    elif shape[1] == 3:
        ret = np.array([rgb_to_hex(x) for x in rgb])
    else:
        raise RuntimeError("Do not understand %s to convert to hex" % rgb)

    return ret


def get_rng_color(n=1, normed=True, color_list=tableau20):
    """Gets random `n` RGB values from tableau20 list"""
    if n > len(color_list):
        raise RuntimeError("len(color_list) !> n [%i !> %i]"
                           % (len(color_list), n))
    idxs = np.random.choice(np.arange(len(color_list)), size=n, replace=False)
    rgbs = tableau20[idxs]
    if normed:
        rgbs = np.divide(rgbs, 255.)

    return rgbs


def get_color_arr(arr, colors='random', normed=True):
    """
    Returns a list of colors corresponding to each element in `arr`.

    Parameters
    ------
    arr: ndarray, iterable
        List of labels or distinguishing elements to assign colors to

    colors: dict, str, default='random'
        Dictionary mapping each unique element in arr to a specific color.
        Else, if `colors=random`, will randomly assign a color to each
        unique element in `arr`.

    normed: bool, default=True
        Whether to return the RGB values within [0, 255], or [0, 1.0]

    Returns
    ------
    color_arr: ndarray of RGB tuples
        Returns an array where each element corresponds to a color within
        `arr`, mapped by unique elements of arr. If normed is True,
        returns RGB values between [0, 1.0], else returns values in [0, 255]
    """
    uniques = np.unique(arr)

    if isinstance(colors, dict):
        # check if all keys of colors are present in unique(arr)
        color_keys = np.sort(colors.keys())
        if not np.all(np.equal(color_keys, uniques)):
            raise RuntimeError("Unique values of `arr` are not all contained"
                               "within keys of `colors`. Found \n"
                               "unique(arr) = %s\n"
                               "colors.keys() = %s"
                               % (list(uniques), list(color_keys)))

    elif colors == 'random':
        rng_colors = get_rng_color(n=len(uniques), normed=normed)
        colors = dict(zip(uniques, rng_colors))
    else:
        raise RuntimeError("Do not recognize colors = %s" % colors)

    color_arr = np.array([colors[x] for x in arr])
    return color_arr
