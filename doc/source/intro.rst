Overview
========

sacc is a schema for storing summary statistic data, metadata, and covariances for the Dark Energy Science Collaboration (DESC).

A sacc file can contain all the observational information required to make theoretical predictions for the mean of a measured quantity, and to calculate a likelihood of it.

Currently sacc files can be saved to a FITS format, but the schema is designed to make it easy to change this if neeed; the structure of the data (in memory) is the focus, rather than the format.


Basic Structure
---------------

A sacc.Sacc object can contain:

- a series of DataPoint objects
- a series of Tracer objects
- a single Covariance object
- additional metadata

Creating Sacc objects
---------------------

A typical workflow for creating new sacc files is:

- instantiate an empty Sacc object with :code:`s = sacc.Sacc()`.
- add an tracer objects that will be used with :code:`s.add_tracer(type_name, tracer_name, ...)`
- one by one, add data points to it in whatever order you prefer with :code:`s.add_data_point(data_type, tracers, value, ...)`
- when finished, add a covariance in the same order with :code:`s.add_covariance(C)`
- save to file using :code:`s.save_fits(filename)`

Reading Sacc objects
--------------------

If you are using a sacc file, for exampe in an MCMC, or for plotting:

- load the sacc data into memory with :code:`s = sacc.Sacc.load_fits(filename)`
- find what data types are in the file with :code:`dts = s.get_data_types()`
- for each data type, find what tracer combinations (e.g. tomographic bin pairs) are available with :code:`tracer_sets = s.get_tracer_combinations(dt)`
- for each pair of tracers, get the mean values with :code:`data = s.get_mean(dt, tracers)`, and, for example, window functions using :code:`windows = s.get_tag(dt, tracers, "window")` or similar for other binning information

You can also select pieces of the data and covariance with various different API methods.

Data Types
----------

Every data point in Sacc has a data type, a string that identifies the type of measurement it refers to.

There are a number of predefined type strings that you can see like this::

    import sacc
    print(sacc.standard_types)


If your data corresponds to one of these types then it's better to use the pre-defined name.
Otherwise, you can make your own.  There is a standard format for these strings::

    {sources}_{properties}_{statistic}[_{subtype}]

where the last item, subtype, is optional.  If there are multiple sources or properties (as in,
for example, cross-correlation measurements) then they are separated by being shown in camelCase.

You can create a type string in the correct format using the command :code:`sacc.build_data_type_name`::

    import sacc
    # the astrophysical sources involved.
    # We use 'cm21' since '21cm' starts with a number which is not allowed in variable names.
    sources = ['quasars', 'cm21']
    # the properties of these two sources we are measuring.  If they were the same
    # property for the two sources we would not repeat it
    properties = ['density', 'Intensity']
    # The statistc, Fourier space C_ell values
    statistic = 'cl'
    # There is no futher specified needed here - everything is scalar.
    subtype = None
    data_type = sacc.build_data_type_name(sources, properties, statistic, subtype)
    print(data_type)
    # prints 'quasarsCm21_densityIntensity_cl'



Data Points
-----------

Each DataPoint object contains:

- a data type string
- a series of strings listing which tracers apply to it
- a value of the data point
- a dictionary of tags, for example describing binning information

Tracers & Windows
-----------------

Different types of tracer each have their own subclass.  For example an NZTracer describes a tomographic bin of sources with a given redshift histogram n(z).

Window functions are stored as a tag on data points, and are represented by Window subclass instances.

Covariance
----------

A single covariance applies to the whole data file.  It can be specified as block

See the API documentation for full details of what is available now.
