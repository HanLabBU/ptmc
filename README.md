# Python Tif Motion Correction
Python package with tools to perform Multipage-Tif Motion Correction of videos.

Written for Python 2.7.

## Installation
Installation and associated requirements can be installed after downloading/cloning using

```python
python setup.py install
or
pip setup.py install
```

## Contains the following Modules

* io - Input/Output functions for loading and saving multipage tiff stacks

* processing - Functions for basic image processing manipulations.  Parallelized versions of the same functions using the Python Multiprocessing package are available in "parallel_processing"

* pipelines - Standardized functions that string together various functions from processing.  Things such as making a reference frame, or correcting an image stack, where certain filters are applied in a particular order.  Parallelized versions of the same functions using the Python Multiprocessing package are available in "parallel_pipelines"
