sacc
====

SACC (Save All Correlations and Covariances) is a format and reference library for general storage
of summary statistic measurements for the Dark Energy Science Collaboration (DESC) within the Large Synoptic
Survey Telescope (LSST) project.


Installation
------------

You can install with the command:

``pip install sacc``

(For local installation you might need to add `--user`)

Or for development versions you can download the repository with git and install from there using ``python setup.py install``

Examples
--------

The examples directory on github contains ipython notebooks showing various ways of constructing sacc data,
manipulating it, saving it, and loading it.

Conceptual Summary
------------------

Sacc models summary statistics using the following concepts:

- a Sacc dataset, containing all the information needed to construct likelihoods of some data.
- Tracers, objects usually corresponding to groups of astrophysical objects and the metadata needed to make predictions for theoretical quantities based on them.
- Windows, objects describing the mapping from a range of theory measurements to individual binned statistics
- Data Points, statistical measurements of some observable quantity, each of which has one or more Tracers and optionally some Windows.
- Covariances, describing the statistic covariance between data points.


Documentation
-------------

Documentation can be found [on ReadTheDocs](https://sacc.readthedocs.io/en/latest/).
