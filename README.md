# sacc

SACC (Save All Correlations and Covariances, an utterly crappy acronym inspired by usually equally bad attempts by David A) is  a format and reference library for general storage of 2-dimensional power spectra and correlation functions and their covariance matrices in the HDF5 format. It is very loosely inspired by Joe Zunz's [2point](https://github.com/joezuntz/2point).

# Quick start

Install by saying

```
./setup.py install
```
For local installation might need to add `--user` to that.
You can create a fake datasets by 
```
./examples/create_sacc.py
```
which you can reload using
```
./examples/load_sacc.py
```
and finally run 
```
./examples/split_sacc.py
```
to load the dataset created by `create_sacc.py`, split it into three files and reload it again for test.

# Conceptual Summary

To describe a generic 2-point correlation function or power spectrum
measurements, one needs several ingredients:

 * *tracer* describes a set of tracers in one photometric bin. The
   tracer description contains the distribution N(z) of tracers and
   possibly some uncertainty in N(z) in terms of templates to
   marginalise over. A different photometric bin will be a different
   tracer, but one can link tracers of the "same kind" (i.e. LSST
   galaxies) by a common ID root in tracer name. We also allow for
   external "tracers" such as CMB kappa measurement.
 * *binning* describes a binning of the power spectrum. In short, it
   is a list of measurements, where each measurement is defined by an
   ell (or separation in case of configuration space), the pair of
   tracers it refers to, the actual quantity measured (e.g. shear or
   numebr density) and the window function. The binning specifies
   what is measured, but not the actual numbers. We can measure auto
   and cross power with different binnings.
 * *mean* is the vector of measurements. It has the same number of
   entries as binning
 * *precision* is the inverse covariance matrix corresponding to mean
   measurements
   
The `sacc` is essentially a container for the tuples of tracers,
binnings, mean vector and precision matrix. 

# Documentation

Thus python module should allow reading anbd writing files in `sacc`
format. If you need to read these files in other context, please see
documentation [in rtd](https://sacc.readthedocs.io/en/latest/) and
[here](doc/format.md).
