# Description of the SACC format

The format has been designed with two goals in mind:
 * flexibilty to store a wide variety of data 
 * realisations that we will need iterations on many realisations, so we need flexibility to carry data vectors or covariances separate from the metadata, since one or both will be reused in reruns of whatever analysis.
 
 The description of some 2point function measurement can be divided into roughly three parts: metadata, mean vector and covariance.
 
## Metadata ##
 
Metadata is stored in a HDF group, typically the root group in the file pointed to by the user. The actual mean vector and precision matrix (C inverse) can be stored in the same group as HDF datasets named "mean" and "precision" respectively and if these do not exist their location can be optional specified by specifying the filename in `mean_file_path` and `precision_file_path` instead. Note that use can always manually override these options. (Instead of "precision" you can also have a "covariance" instead (and then store C), but not both. [not yet implemented])

### Tracers ###

Information on tracers used is stored in HDF group `tracers`. This group must have an attribute `tracer_list` which is a list of strings defining tracer names. Ordering is important, because index in this list defines the index by which tracers are referred to later. For each tracer name there must be a subdataset inside `tracers` group which defines this tracer. 

A tracer datasets must have an attribute `type` which must be of one of the following values:
 * `spin0`: e.g. number counts, kappa, magnification, CMB temperature
 * `spin2`: e.g. CMB polarization, lensing shear
 * (other we'll add as needs arise)
 
If it is CMB or CMB lensing, no further info is required (it is assumed that e.g. reconstruction noise, resolution, etc are all corrected for and reflected in the errorbars). For tracer, the datasets must contain a at least 2D array with fiels `z` and `Nz` (with obvious meaning). The datset can optionally contain columns with fiducial values of bias (`b`) and anything else you might want to store (to help checking mocks with results, etc.)

Uncertainities in N(z) can be presently described using a set of zero average (but not neccessarily orthogonal) vectors stored in columns `DNz_0`, `DNz_1`, etc. Their amplitude should correspond to their variance. In short, the covariance error on Nz is given as sum vv^T, where sum is over v=`DNz_0`, `DNz_1`,... Additionally, overall uncertainties expressed in terms of translating and stretching Nz should be in attributes `Nz_sigma_logmean` and `Nz_sigma_logwidth` where the translating variable is 1+z or equivalently (up to a sign) a=1/(1+z).

Often many 'tracers' in this sense will really be the same parent population chopped into pieces. In that case each tracer can optionally carry a `exp_sample`, which is an identifying string tying the same tracers together. E.g. all LSST galaxies will carry `exp_sample="lsst_gal"` and those from red magic will have `exp_sample="lsst_redm"`.

### Binning / Indices ###

Information of which index in the mean/precision matrix correspond to
which correlation is stored in the dataset named `binning`, which also
must exist. It is a table with the following columns:
 * `type`: 2-letter value. Use `FF` for Fourier/harmonic-space measurements. In configuration space, use `+R` and `+C` for the real (EE+BB) and complex (BE+EB) parts of the `+` correlation and `-R` or `-C` for the `-` correlation function (corresponding to the EE-BB and BE-EB terms respectively). For 1-pt functions (e.g., cluster counts), use `+N`.
 * `ls`: value of ell or separation (even when we have windows, for e.g. plotting)
 * `T1`: index of tracer 1 defined above
 * `Q1`: quantity from tracer 1. Use `S` for scalar (e.g. temperature, number counts, lensing convergence), `E`/`B` for spin-2 (e.g. CMB polarization and WL).
 * `T2`: index of tracer 2 defined above
 * `Q2`: quantity from tracer 1
 
It can also optionally have the following fields:
 * `window` : specifying index of the window defined above, or -1 if no window.
 * `Delta ls` : specifying window width assuming top-hat windows
 * `error`: actual diagonal error (for plotting)

If any of the `type==C` need to also specify attribute `sunit` (a
string with unit for separation)

This uniquely defines that each datapoint in the mean vector means. 

### Windows [not yet implemented] ###

Window functions are optional (see below). Window functions are stored in an HDF group `windows`. Inside this groups there are small datasets named as numbers starting with 0.  Each such datasets need to a 2D array of `l` and `w` or `s` and `w` (for configuration space). Units of `s` need not be specified -- they will be defined in the data vector. These windows specify how Cl needs to be integrated over l in order to produce measurement.

## Mean values ##

Mean values are stored in a HDF datasets called `mean` which might
exist inside the same HDF5 as metadata, but not necessarily so (see
`mean_file_path` comment above)

It is a single 1D vector for floats whose meaning is defined by
`indices` dataset above. 
 
 
## Precision matrix ##
 
Precision matrix defined the NxN matrix corresponding to Cinverse to the data vector of size N. It is a HDF dataset called `precision`, which might exist inside the same HDF5 as metadata, but not neccessarily so. It must have an attribute called `type` which must be one of the following:
 * `diagonal` : A vector of size `N` is stored
 * `ell_block_diagonal` : A double loop over non-repeating elements of C, storing just those points where ell/s values match. 
 * `dense`: Just a full 2D monty.
 
 
