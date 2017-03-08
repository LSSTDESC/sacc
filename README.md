# sacc

SACC (Save All Correlations and Covariances, an utterly crappy acronym inspired by usually equally bad attempts by David A) is  a format and reference library for general storage of 2-dimensional power spectra and correlation functions and their covariance matrices in the HDF5 format. It is very loosely inspired by Joe Zunz's [2point](https://github.com/joezuntz/2point).

# Description of the format

The format has been designed with two goals in mind:
 * flexibilty to store a wide variety of data 
 * realisations that we will need iterations on many realisations, so we need flexibility to carry data vectors or covariances separate from the metadata, since one or both will be reused in reruns of whatever analysis.
 
 The description of some 2point function measurement can be divided into roughly three parts: metadata, mean vector and covariance.
 
## Metadata ##
 
Metadata is stored in a HDF group, typically the root group in the file pointed to by the user. The actual mean vector and precision matrix (C inverse) can be stored in the same group as HDF datasets named "mean" and "precision" respectively and if these do not exist their location can be optional specified by creating an empty dataset with the same name and attaching an attribute `file_path` instead. Note that use can always manually override these options. Instead of "precision" you can also have a "covariance" instead (and then store C), but not both.

### Tracers ###

Information on tracers used is stored in HDF group `tracers`. This group must have an attribute `tracer_list` which is a list of strings defining tracer names. Ordering is important, because index in this list defines the index by which tracers are referred to later. For each tracer name there must be a subdataset inside `tracers` group which defines this tracer. 

A tracer datasets must have an attribute `type` which must be of one of the following values:
 * `point`, i.e galaxies 
 * `cmb`, for CMB (for e.g. ISW studies)
 * `cmbl`, for CMB kappa reconstruction
 * (other we'll add as needs arise)
 
If it is CMB or CMB lensing, no further info is required (it is assumed that e.g. reconstruction noise, resolution, etc are all corrected for and reflected in the errorbars). For tracer, the datasets must contain a at least 2D array with fiels `z` and `Nz` (with obvious meaning). The datset can optionally contain columns with fiducial values of bias (`b`) and anything else you might want to store.

[TBC]

