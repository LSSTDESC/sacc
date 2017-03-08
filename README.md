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
 * `cmb`, for CMB (for e.g. ISW studies, WL, etc (note that CMB kappa vs CMB primary will be distinguished below)
 * (other we'll add as needs arise)
 
If it is CMB or CMB lensing, no further info is required (it is assumed that e.g. reconstruction noise, resolution, etc are all corrected for and reflected in the errorbars). For tracer, the datasets must contain a at least 2D array with fiels `z` and `Nz` (with obvious meaning). The datset can optionally contain columns with fiducial values of bias (`b`) and anything else you might want to store (to help checking mocks with results, etc.)

Uncertainities in N(z) can be presently described using a set of zero average (but not neccessarily orthogonal) vectors stored in columns `DNz_0`, `DNz_1`, etc. Their amplitude should correspond to their variance. In short, the covariance error on Nz is given as sum vv^T, where sum is over v=`DNz_0`, `DNz_1`,... Additionally, overall uncertainties expressed in terms of translating and stretching Nz should be in attributes `Nz_sigma_logmean` and `Nz_sigma_logwidth` where the translating variable is 1+z or equivalently (up to a sign) a=1/(1+z).

Often many 'tracers' in this sense will really be the same parent population chopped into pieces. In that case each tracer can optionally carry a `exp_sample`, which is an identifying string tying the same tracers together. E.g. all LSST galaxies will carry `exp_sample="lsst_gal"` and those from red magic will have `exp_sample="lsst_redm"`.



### Windows ###

Window functions are optional (see below). Window functions are stored in an HDF group `windows`. Inside this groups there are small datasets named as numbers starting with 0.  Each such datasets need to a 2D array of `l` and `w` or `s` and `w` (for configuration space). Units of `s` need not be specified -- they will be defined in the data vector. These windows specify how Cl needs to be integrated over l in order to produce measurement.

## Mean values ##

Mean values are stored in a HDF datasets called `mean` which might exist inside the same HDF5 as metadata, but not neccessarily so. If it contains a single config space measurement, it must also have an attribute `sunit` to specify angular units (what are valid units can be up to final user code). It is a 2D dataset which must have the following fields:
 * `type`: Letter `F`/`C` for Fourier/Configuration space measurements. Can later add more letters for compensated measured. 
 * `ls`: value of ell or separation (even when we have windows, for e.g. plotting)
 * `T1`: index of tracer 1 defined above
 * `Q1`: quantity from tracer 1. Use `I` for intenstiy, `E`/`B` for CMB polarization and WL, `P` for point sources, `+`/`-` for corresponding WL correlation funcs, `K` for WL kappa, might need to invent more sources.
 * `T2`: index of tracer 2 defined above
 * `Q2`: quantity from tracer 1
 * `value` : actual mean mesurement
 * `error`: actual diagonal error (for plotting)
 
It can also have the following fields:
 * `window` : specifying index of the window defined above, or -1 if no window.
 * `Delta ls` : specifying window width assuming top-hat windows
 
## Precision matrix ##
 
Precision matrix defined the NxN matrix corresponding to Cinverse to the data vector of size N. It is a HDF dataset called `precision`, which might exist inside the same HDF5 as metadata, but not neccessarily so. It must have an attribute called `type` which must be one of the following:
 * `diagonal` : A vector of size `N` is stored
 * `ell_block_diagonal` : A double loop over non-repeating elements of C, storing just those points where ell/s values match. 
 * `dense`: Just a full 2D monty.
 
 


