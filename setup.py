import os
import re
from setuptools import setup
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str
    with open(os.path.join(here, 'itsne', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search("__version__ = '(.*)'", init_py).groups()[0]
except Exception:
    version = ''

setup(name='itsne',
      version=version,
      description='Interactive visualizations of data using t-SNE clustering with Bokeh',
      author='Long Van Ho',
      author_email='longvho916@gmail.com',
      url='https://github.com/longubu/itsne',
      download_url='---',
      license='MIT',
      extras_require={
          '': [''],
      },
      packages=find_packages())
