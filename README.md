# itsne
Interactive visualization of data using t-SNE and bokeh.

itsne purpose:
- Quick interactive visualization of high-dimensional datasets for in-depth analysis and insights of the data, algorithm, and feature representation.
- and it's pretty (can use a little more work)

# Installation
## Dependencies

- Bokeh
- Numpy
- PIL
- sklearn

If you don't already have these packages installed, it is
recommended to install via third party distribution such as anaconda. If you're using linux, it's recommended to install using the package manager -- else it may do a lengthy build process with many depedencies and may lead to a much slower configuration. 

## Build from source (does not include dependencies)

	git clone https://github.com/longubu/itsne.git
	cd itsne
	python setup.py install
	
# Usage
See `examples/*` for usage.
- [mnist_imgs.py](examples/mnist_imgs.py): Example of using the itsne for clustering raw-pixels from MNIST dataset, plotting a subset of data using the images. See [output](examples/outputs/mnist_imgs.html)
- [mnist_circles.py](examples/mnist_circles.py): Example of using the itsne for clustering raw-pixels from MNIST dataset, plotting the circles. See [output](examples/outputs/mnist_circles.html)
- Many more examples to come...

# TODO

- More examples
- Easier API, more flexibility
- Better looking visualization
