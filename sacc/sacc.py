import copy
import os
import re
import warnings

from astropy.io import fits
from astropy.table import Table

# Module-level constants for Sacc file format versions
SACCFVER = 2  # Current FITS version
SACCHDF5VER = 1  # Current HDF5 version
import numpy as np

from .tracers import BaseTracer
from .windows import BandpowerWindow
from .covariance import BaseCovariance, concatenate_covariances
from .utils import unique_list
from .data_types import standard_types, DataPoint
from . import io

class Sacc:
    """
    A class containing a selection of LSST summary statistic measurements,
    their covariance, and the metadata necessary to compute theoretical
    predictions for them.
    """

    def __init__(self):
        """
        Create an empty data  set ready to be built up
        """
        self.data = []
        self.tracers = {}
        self.covariance = None
        self.metadata = {}
        self.tracer_uncertainties = {}

    def __len__(self):
        """
        Return the number of data points in the data set.

        Returns
        -------
        n: int
            The number of data points
        """
        return len(self.data)

    def __eq__(self, other):
        """
        Test for equality between two Sacc instances.

        Checks whether the two values are equal.  This is a
        complete equality check, and will check that the data points,
        tracers, covariance and metadata are all the same.

        Parameters
        ----------
        other: Sacc instance
            The other data set to compare with

        Returns
        -------
        equal: bool
            True if the two data sets are the same, False otherwise.
        """
        if not isinstance(other, Sacc):
            return False

        if self.data != other.data:
            return False

        if len(self.tracers) != len(other.tracers):
            return False
        if set(self.tracers.keys()) != set(other.tracers.keys()):
            return False
        for k1, v1 in self.tracers.items():
            v2 = other.tracers[k1]
            if not v1 == v2:
                return False

        if self.covariance != other.covariance:
            return False

        if self.metadata != other.metadata:
            return False

        return True


    def copy(self):
        """
        Create a copy of the data set with no data shared with the original.
        You can safely modify the copy without it affecting the original.

        Returns
        -------
        S: Sacc instance
            A new instance of the data set.
        """
        return copy.deepcopy(self)

    def to_canonical_order(self):
        """
        Re-order the data set in-place to a standard ordering.
        """

        # Define the ordering to be used
        # We need a key function that will return the
        # object that python's default sorted function will use.
        def order_key(row):
            # Put data types in the order in allowed_types.
            # If not present then just use the hash of the data type.
            if row.data_type in standard_types:
                dt = standard_types.index(row.data_type)
            else:
                dt = hash(row.data_type)
            # If known, order by ell or theta.
            # Otherwise just use whatever we have.
            if 'ell' in row.tags:
                return (dt, row.tracers, row.tags['ell'])
            if 'theta' in row.tags:
                return (dt, row.tracers, row.tags['theta'])
            return (dt, row.tracers, 0.0)
        # This from
        # https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
        indices = [i[0] for i in sorted(enumerate(self.data),
                                        key=lambda x:order_key(x[1]))]

        # Assign the new order.
        self.reorder(indices)

    def reorder(self, indices):
        """
        Re-order the data set in-place according to the indices passed in.

        If not all indices are included in the input then the data set will
        be cut down.

        Parameters
        ----------
        indices: integer list or array
            Indices for the re-ordered data
        """
        self.data = [self.data[i] for i in indices]

        if self.has_covariance():
            self.covariance = self.covariance.keeping_indices(indices)

    #
    # Builder methods for building up Sacc data from scratch in memory
    #

    def add_tracer(self, type_name, name,
                   *args, **kwargs):
        """
        Add a new tracer

        Parameters
        ----------
        type_name: str
            A string corresponding to one of the known tracer types,
            or 'misc' to use a new tracer with no parameters.
            e.g. "NZ" for n(z) tracers

        name: str
            A name for the tracer

        *args:
            Additional arguments to pass to the tracer constructor.
            These depend on the type of the tracer.  For n(z) tracers
            these should be z and nz arrays

        **kwargs:
            Additional keyword arguments to pass to the tracer constructor.
            These depend on the type of the tracer.  There are no
            kwargs for n(z) tracers

        Returns
        -------
        None

        """
        tracer = BaseTracer.make(type_name, name,
                                 *args, **kwargs)
        self.add_tracer_object(tracer)

    def add_tracer_object(self, tracer):
        """
        Add a pre-constructed BaseTracer instance to this data set.
        If you just have, for example the z and n(z) data then
        use the add_tracer method instead.

        Parameters
        ----------
        tracer: Tracer instance
            The tracer object to add to the data set
        """
        self.tracers[tracer.name] = tracer

    def add_tracer_uncertainty_object(self, uncertainty):
        """
        Add a pre-constructed tracer uncertainty object to this data set.

        Parameters
        ----------
        uncertainty: BaseTracerUncertainty instance
            The uncertainty object to add to the data set
        """
        self.tracer_uncertainties[uncertainty.name] = uncertainty

    def add_data_point(self, data_type, tracers, value,
                       tracers_later=False, **tags):
        """
        Add a data point to the set.

        Parameters
        ----------
        data_type: str

        tracers: tuple of str
            Strings corresponding to zero or more of the tracers
            in the data set. These should either be already set up
            using the add_tracer method, or you could set
            tracers_later=True if you want to add them later.
            e.g. for 2pt measurements the tracers are the names of
            the two n(z) samples.

        value: float
            A single value for the data point

        tracers_later: bool
            If True, do not complain if the tracers are not know
            already.

        **tags:
            Tags to apply to this data point.
            Tags can be any arbitrary metadata that you might want later,
            For 2pt data the tag would include an angle theta or ell.

        Returns
        -------
        None
        """
        if self.has_covariance():
            raise ValueError("You cannot add a data point after "
                             "setting the covariance")
        tracers = tuple(tracers)
        for tracer in tracers:
            if (tracer not in self.tracers) and (not tracers_later):
                raise ValueError(f"Tracer named '{tracer}' is not in the "
                                 "known list of tracers. "
                                 "Either put it in before adding data "
                                 "points or set tracers_later=True")
        d = DataPoint(data_type, tracers, value, **tags)
        self.data.append(d)

    def add_covariance(self, covariance, overwrite=False):
        """
        Once you have finished adding data points, add a covariance
        for the entire set.

        Parameters
        ----------
        covariance: array or list
            2x2 numpy array containing the covariance of the added data points
            OR a list of blocks
        overwrite: bool
            If True, it overwrites the stored covariance matrix with the given
            one.

        Returns
        -------
        None
        """
        if self.has_covariance() and not overwrite:
            raise RuntimeError("This sacc file already contains a covariance"
                               "matrix. Use overwrite=True if you want to "
                               "replace it for the new one")

        if isinstance(covariance, BaseCovariance):
            cov = covariance
        else:
            cov = BaseCovariance.make(covariance)

        expected_size = len(self)
        if not cov.size == expected_size:
            raise ValueError("Covariance has the wrong size. "
                             f"Should be {expected_size} but is {cov.size}")

        self.covariance = cov

    def has_covariance(self):
        """ Return whether or not this data set has a covariance attached to it

        Returns
        -------
        bool
            Whether or not a covariance has been added to this data
        """
        return self.covariance is not None

    def _indices_to_bool(self, indices):
        # Convert an array of indices into a boolean True mask
        if indices.dtype not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError(f"Wrong indices type ({indices.dtype}) - "
                             "expected integers or boolean")
        m = np.zeros(len(self), dtype=bool)
        for i in indices:
            m[i] = True
        return m

    def keep_indices(self, indices):
        """
        Select data points, keeping only values where the mask is True or
        an index is included in it.

        You can use Sacc.remove_indices to do the opposite operation,
        keeping points where the mask is False.

        You use the Sacc.keep_selection method to find indices and apply
        this method automatically, or the Sacc.indices method to manually
        select indices.

        Parameters
        ----------
        indices: array or list
            Mask must be either a boolean array or a list/array of integer
            indices to remove. If boolean then True means to keep a data
            point and False means to cut it if integers then values
            indicate data points to keep.
        """
        indices = np.array(indices)

        # Convert integer masks to booleans
        if indices.dtype != bool:
            indices = self._indices_to_bool(indices)

        self.data = [d for i, d in enumerate(self.data) if indices[i]]
        if self.has_covariance():
            self.covariance = self.covariance.keeping_indices(indices)

    def remove_indices(self, indices):
        """
        Remove data points, getting rid of points where the mask is True
        or an index is included in it.

        You can use Sacc.keep_indices to do the opposite operation,
        keeping points where the mask is True.

        You use the Sacc.remove_selection method to find indices and
        apply this method automatically, or the Sacc.indices method to
        manually select indices.

        Parameters
        ----------
        indices: array or list
            Mask must be either a boolean array or a list/array of
            integer indices to remove. If boolean then True means
            to cut data point and False means to keep it if integers
            then values indicate data points to cut out
        """
        indices = np.array(indices)

        # Convert integer masks to booleans
        if indices.dtype != bool:
            indices = self._indices_to_bool(indices)

        # Get the mask method to do the actual work
        self.keep_indices(~indices)

    def indices(self, data_type=None, tracers=None, warn_empty=True, **select):
        """
        Find the indices of all points matching the given selection criteria.

        Parameters
        ----------
        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold

        Returns
        indices: array
            Array of integer indices of matching data points

        """
        indices = []
        if tracers is not None:
            tracers = tuple(tracers)

        # Look through all data points we have
        for i, d in enumerate(self.data):
            # Skip things with the wrong type or tracer
            if not ((tracers is None) or (d.tracers == tracers)):
                continue
            if not (data_type is None or d.data_type == data_type):
                continue
            # Remove any objects that don't match the required tags,
            # including the fact that we can specify tag__lt and tag__gt
            # in order to remove/accept ranges
            ok = True
            for name, val in select.items():
                if name.endswith("__lt"):
                    name = name[:-4]
                    if not d.get_tag(name) < val:
                        ok = False
                        break
                elif name.endswith("__gt"):
                    name = name[:-4]
                    if not d.get_tag(name) > val:
                        ok = False
                        break
                else:
                    if not d.get_tag(name) == val:
                        ok = False
                        break
            # Record this index
            if ok:
                indices.append(i)

        if len(indices) == 0 and warn_empty:
            if tracers is None:
                warnings.warn("Empty index selected")
            else:
                warnings.warn("Empty index selected - maybe you "
                              "should check the tracer order?")

        return np.array(indices, dtype=int)

    def remove_selection(self, data_type=None, tracers=None,
                         warn_empty=True, **select):
        """
        Remove data points, getting rid of points matching the given criteria.

        You can use Sacc.keep_selection to do the opposite operation, keeping
        points where the criteria are matched.

        You can manually remove points using the Sacc.indices and
        Sacc.remove_indices methods.

        Parameters
        ----------
        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold
        """

        indices = self.indices(data_type=data_type, tracers=tracers,
                               warn_empty=warn_empty, **select)
        self.remove_indices(indices)

    def keep_selection(self, data_type=None, tracers=None,
                       warn_empty=True, **select):
        """
        Remove data points, keeping only points matching the given criteria.

        You can use Sacc.remove_selection to do the opposite operation,
        keeping points where the criteria are not matched.

        You can manually remove points using the Sacc.indices and
        Sacc.keep_indices methods.

        Parameters
        ----------
        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold
        """
        indices = self.indices(data_type=data_type, tracers=tracers,
                               warn_empty=warn_empty, **select)
        self.keep_indices(indices)

    def _get_tags_by_index(self, tags, indices):
        """
        Get the value of a one or more named tags for (a subset of) the data.

        Parameters
        ----------

        tags: list of str
            Tags to look up on the selected data

        indices: list or array
            Indices of data points

        Returns
        -------
        values: list of lists
            For each input tag, a corresponding list of the value of that
            tag for given selection, in the order the matching data points
            were added.


        """
        indices = set(indices)
        values = [[d.get_tag(tag)
                   for i, d in enumerate(self.data) if i in indices]
                  for tag in tags]
        return values

    def get_tags(self, tags, data_type=None, tracers=None, **select):
        """
        Get the value of a one or more named tags for (a subset of) the data.

        Parameters
        ----------

        tags: list of str
            Tags to look up on the selected data

        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold

        Returns
        -------
        values: list of lists
            For each input tag, a corresponding list of the value of
            that tag for given selection, in the order the matching
            data points were added.
        """
        indices = self.indices(data_type=data_type,
                               tracers=tracers, **select)
        return self._get_tags_by_index(tags, indices)

    def get_tag(self, tag, data_type=None, tracers=None, **select):
        """
        Get the value of a one tag for (a subset of) the data.

        Parameters
        ----------
        tag: str
            Tag to look up on the selected data

        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold

        Returns
        -------
        values: list
            A list of the value of the tag for given selection,
            in the order the matching data points were added.
        """
        return self.get_tags([tag], data_type=data_type,
                             tracers=tracers, **select)[0]

    def get_data_points(self, data_type=None, tracers=None, **select):
        """
        Get data point objects for a subset of the data

        Parameters
        ----------
        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold

        Returns
        -------
        values: list
            A list of the data point objects for the selection,
            in the order they were added.
        """
        indices = self.indices(data_type=data_type, tracers=tracers, **select)
        return [self.data[i] for i in indices]

    def get_mean(self, data_type=None, tracers=None, **select):
        """
        Get mean values for each data point matching the criteria.

        Parameters
        ----------

        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold

        Returns
        -------
        values: list
            The mean values for each matching data point,
            in the order they were added.

        """
        indices = self.indices(data_type=data_type, tracers=tracers, **select)
        return self.mean[indices]

    def get_standard_deviation(self, data_type=None, tracers=None, **select):
        """
        Get standard deviation values for each data point matching the criteria.

        This requires the covariance matrix to be set.

        Parameters
        ----------

        data_type: str
            Select only data points which are of this data type.
            If None (the default) then match any data types

        tracers: tuple
            Select only data points which match this tracer combination.
            If None (the default) then match any tracer combinations.

        **select:
            Select only data points with tag names and values matching
            all values provided in this kwargs option.
            You can also use the syntax name__lt=value or
            name__gt=value in the selection to select points
            less or greater than a threshold

        Returns
        -------
        values: array
            The standard deviation values for each matching data point,
            in the order they were added.

        """
        indices = self.indices(data_type=data_type, tracers=tracers, **select)
        return np.sqrt(self.covariance.get_block(indices).diagonal())

    def get_data_types(self, tracers=None):
        """
        Get a list of the different data types stored in the Sacc

        Parameters
        ----------
        tracers: tuple
            Select only data types which match this tracer combination.
            If None (the default) then match any tracer combinations.

        Returns
        --------
        data_types: list of strings
            A list of the string data types in the data set
        """
        data_types = unique_list(d.data_type for d in self.data if
                                 ((tracers is None) or (d.tracers == tracers)))

        return data_types

    def has_tracer(self, name):
        """
        Determine whether a tracer object with the given name is present

        Parameters
        ----------
        name: str
            A string name of a tracer

        Returns
        -------
        value: True if the tracer exists, else False
        """
        return name in self.tracers

    def get_tracer(self, name):
        """
        Get the tracer object with the given name

        Parameters
        -----------
        name: str
            A string name of a tracer

        Returns
        -------
        tracer: BaseTracer object
            The object corresponding to the name.
        """
        return self.tracers[name]

    def get_tracer_combinations(self, data_type=None):
        """
        Find all the tracer combinations (e.g. tomographic bin pairs)
        for the given data type

        Parameters
        -----------
        data_type: str
            A string name of the data type to find

        Returns
        -------
        combinations: list of tuples of strings
            A list of all the tracer combinations found
            in any data point.  No specific ordering.
        """
        indices = self.indices(data_type=data_type)
        return unique_list(self.data[i].tracers for i in indices)

    def remove_tracers(self, names):
        """
        Remove the tracer objects and their associated data points

        Parameters
        -----------
        names: list
            A list of string names of the tracers to be removed

        """

        for trs in self.get_tracer_combinations():
            for tri in trs:
                if tri in names:
                    self.remove_selection(tracers=trs)
                    break

        for name in names:
            del self.tracers[name]

    def keep_tracers(self, names):
        """
        Keep only the tracer objects and their associated data points.

        Parameters
        -----------
        names: list
            A list of string names of the tracers to be kept

        """

        for trs in self.get_tracer_combinations():
            for tri in trs:
                if tri not in names:
                    self.remove_selection(tracers=trs)
                    break

        trs_names = list(self.tracers.keys())
        for name in trs_names:
            if name not in names:
                del self.tracers[name]

    def rename_tracer(self, name, new_name):
        """
        Get the tracer object with the given name

        Parameters
        -----------
        name: str
            A string name of a tracer to be changed the name
        new_name: str
            A string with the new name of the tracer

        """

        tr = self.tracers.pop(name)
        tr.name = new_name
        self.tracers[new_name] = tr

        for d in self.data:
            new_trs = []
            for tri in d.tracers:
                if tri == name:
                    tri = new_name

                new_trs.append(tri)

            d.tracers = tuple(new_trs)

    @property
    def mean(self):
        """
        Get the vector of mean values for the entire data set.

        Returns
        -------
        mean: array
            numpy array with all the mean values in the data set
        """
        return np.array([d.value for d in self.data])

    @mean.setter
    def mean(self, mu):
        """
        Set the vector of mean values for the entire data set.

        Parameters
        -----------

        mu: array
            Replace the mean values of all the data points.
        """
        if not len(mu) == len(self.data):
            raise ValueError("Tried to set mean with thing of length {}"
                             " but data is length {}".format(len(mu),
                                                             len(self.data)))
        for m, d in zip(mu, self.data):
            d.value = m

    def _make_window_tables(self):
        # Convert any window objects in the data set to tables,
        # and record a mapping from those objects to table references
        # This could easily be extended to other types
        windows = []
        for d in self.data:
            w = d.get_tag("window")
            if w is not None:
                windows.append(w)

        windows = unique_list(windows)
        window_ids = {id(w):w for w in windows}
        return window_ids

    def to_tables(self):
        """
        Convert this data set to a collection of astropy tables.

        Parameters
        ----------
        None

        Returns
        -------
        tables: list of astropy Table objects
            A list of tables, each corresponding to a different
            type of object in the data set.  The tables will have
            metadata that can be used to reconstruct the data set.
        """
        # Get the tracers
        objects = {
            "tracer": self.tracers,
            "data": self.data,
            "window": self._make_window_tables(),
            "metadata": self.metadata,
            "traceruncertainty": self.tracer_uncertainties,
        }

        if self.has_covariance():
            # For now the name will just be "cov", but in future
            # we may support alternatives.
            objects["covariance"] = {self.covariance.name: self.covariance}

        tables = io.to_tables(objects)

        return tables

    def save_fits(self, filename, overwrite=False):
        """
        Save this data set to a FITS format Sacc file.

        Parameters
        ----------
        filename: str
            Destination FITS file name

        overwrite: bool
            If False (the default), raise an error if the file already exists
            If True, overwrite the file silently.
        """

        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"File {filename} already exists. "
                                  "Use overwrite=True to overwrite it.")

        tables = self.to_tables()

        # Add the EXTNAME metadata value to each table.
        # This is used to set the HDU name in the FITS file.
        for table in tables:
            typ = table.meta['SACCTYPE']
            name = table.meta['SACCNAME']
            if typ != 'data':
                cls = table.meta['SACCCLSS']
                extname = f'{typ}:{cls}:{name}'
                table.meta['EXTNAME'] = extname

        # Create the actual fits object
        primary_header = fits.Header()
        primary_header['SACCFVER'] = SACCFVER
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)
            hdus = [fits.PrimaryHDU(header=primary_header)] + [fits.table_to_hdu(table) for table in tables]
        hdu_list = fits.HDUList(hdus)
        io.astropy_buffered_fits_write(filename, hdu_list)

    @classmethod
    def load_fits(cls, filename):
        """
        Load a Sacc data set from a FITS file.

        Don't try to make these FITS files yourself - use the tools
        provided in this package to make and save them.

        Parameters
        ----------
        filename: str
            A FITS format sacc file
        """
        cov = None
        metadata = None
        fitsver = None

        with fits.open(filename, mode="readonly") as f:
            tables = []
            for idx, hdu in enumerate(f):
                if hdu.name.lower() == 'primary':
                    header = hdu.header
                    fitsver = header.get('SACCFVER', None)
                    if fitsver is None:
                        fitsver = 1
                    if fitsver > SACCFVER:
                        raise RuntimeError(f"Unsupported SACC FITS version: {fitsver}")
                    if "NMETA" in header:
                        metadata = {}
                        n_meta = header['NMETA']
                        for i in range(n_meta):
                            k = header[f'KEY{i}']
                            v = header[f'VAL{i}']
                            metadata[k] = v
                elif hdu.name.lower() == 'covariance':
                    cov = BaseCovariance.from_hdu(hdu)
                else:
                    tables.append(Table.read(hdu))

        if metadata is not None:
            tables.append(io.metadata_to_table(metadata))

        # Pass version to from_tables if needed (future-proofing)
        return cls.from_tables(tables, cov=cov)

    def save_hdf5(self, filename, overwrite=False, compression='gzip', compression_opts=4):
        """
        Save this data to a HDF5 format Sacc file.

        Parameters
        ----------
        filename: str
            Destination HDF5 file name
        overwrite: bool
            If False (the default), raise an error if the file already exists
            If True, overwrite the file silently.
        compression: str, optional
            Compression filter to use ('gzip', 'lzf', 'szip', or None). Default is 'gzip'.
        compression_opts : int, optional
            Compression level (0-9 for gzip, where 0 is no compression and 9 is maximum).
            Default is 4 (moderate compression).
        """
        import h5py
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"File {filename} already exists. "
                                  "Use overwrite=True to overwrite it.")
        tables = self.to_tables()

        # Add the EXTNAME metadata value to each table.
        for table in tables:
            typ = table.meta['SACCTYPE']
            name = table.meta['SACCNAME']
            if typ != 'data':
                cls = table.meta['SACCCLSS']
                extname = f'{typ}:{cls}:{name}'
                table.meta['EXTNAME'] = extname

        with h5py.File(filename, 'w') as f:
            # Write version dataset
            f.create_dataset('sacc_hdf5_version', data=np.array([SACCHDF5VER], dtype='i4'))
            used_names = {}
            for table in tables:
                # Build a meaningful dataset name
                typ = table.meta.get('SACCTYPE', 'unknown')
                name = table.meta.get('SACCNAME', None)
                cls = table.meta.get('SACCCLSS', None)
                part = table.meta.get('SACCPART', None)

                # Compose base dataset name
                if typ == 'data' and name:
                    dset_name = f"data/{name}"
                elif typ == 'tracer' and name:
                    dset_name = f"tracer/{name}"
                elif typ == 'traceruncertainty' and name:
                    dset_name = f"traceruncertainty/{name}"
                elif typ == 'window' and name:
                    dset_name = f"window/{name}"
                    if part:
                        dset_name += f"_{part}"
                elif typ == 'covariance' and name:
                    dset_name = f"covariance_{name}"
                elif typ == 'metadata':
                    dset_name = "metadata"
                elif name:
                    dset_name = f"{typ}_{name}"
                else:
                    dset_name = typ

                # Ensure uniqueness by appending an index if needed
                base_name = dset_name
                idx = used_names.get(base_name, 0)
                while dset_name in f:
                    idx += 1
                    dset_name = f"{base_name}_{idx}"
                used_names[base_name] = idx

                table.write(f,
                            path=dset_name,
                            serialize_meta=False,
                            compression=compression,
                            compression_opts=compression_opts
                            )

    @classmethod
    def load_hdf5(cls, filename):
        """
        Load a Sacc object from an HDF5 file.

        Parameters
        ----------
        filename: str
            Path to the HDF5 file.

        Returns
        -------
        sacc_obj: Sacc
            A Sacc object reconstructed from the tables in the HDF5 file.
        """
        import h5py
        recovered_tables = []
        hdf5ver = None
        with h5py.File(filename, 'r') as f:
            # Check version
            if 'sacc_hdf5_version' in f:
                hdf5ver = int(np.array(f['sacc_hdf5_version'])[0])
            else:
                hdf5ver = 1
            if hdf5ver > SACCHDF5VER:
                raise RuntimeError(f"Unsupported SACC HDF5 version: {hdf5ver}")
            # Read all datasets (not groups) in the order they appear
            for key in f.keys():
                if key == 'sacc_hdf5_version':
                    continue
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    table = Table.read(f, path=key)
                    recovered_tables.append(table)
                elif isinstance(item, h5py.Group):
                    for subkey in item.keys():
                        subitem = item[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            table = Table.read(item, path=f"{subkey}")
                            recovered_tables.append(table)
        sacc_obj = cls.from_tables(recovered_tables)
        return sacc_obj

    @classmethod
    def from_tables(cls, tables, cov=None):
        """
        Reassmble a Sacc object from a collection of tables.

        Parameters
        ----------
        objs: dict[str, dict[str, BaseIO]]
            A dictionary of objects, with some of 'tracer', 'data', 'window',
            and 'covariance'. Each key maps to a list of objects
            or a single object.
        """
        s = cls()

        objs =  io.from_tables(tables)

        # Add all the tracers
        tracers = objs.get('tracer', {})
        for tracer in tracers.values():
            s.add_tracer_object(tracer)

        # Add the actual data points. The windows and any future
        # objects that are attached to individual data points
        # will be included in the data points themselves, there is
        # no need to add them separately.
        data = fix_data_ordering(objs.get('data', []))
        for d in data:
            s.data.append(d)

        # Add the covariance, if it is present.
        if "covariance" in objs:
            if cov is not None:
                raise ValueError("Found both a legacy covariance and a new one in the same file.")
            cov = objs["covariance"]["cov"]

        # copy in metadata
        s.metadata.update(objs.get('metadata', {}))

        if cov is not None:
            s.add_covariance(cov)

        for uncertainty in objs.get('traceruncertainty', {}).values():
            s.add_tracer_uncertainty_object(uncertainty)

        return s


    #
    # Methods below here are helper functions for specific types of data.
    # We can add more of them as it becomes clear what people need.
    #
    #

    def _get_2pt(self, data_type, tracer1, tracer2, return_cov,
                 angle_name, return_ind=False):
        # Internal helper method for get_ell_cl and get_theta_xi
        ind = self.indices(data_type, (tracer1, tracer2))

        mu = np.array(self.mean[ind])
        angle = np.array(self._get_tags_by_index([angle_name], ind)[0])

        if return_cov:
            if not self.has_covariance():
                raise ValueError("This sacc data does not have "
                                 "a covariance attached")
            cov_block = self.covariance.get_block(ind)
            if return_ind:
                return angle, mu, cov_block, ind
            return angle, mu, cov_block
        if return_ind:
            return angle, mu, ind
        return angle, mu

    def get_bandpower_windows(self, indices):
        """
        Returns bandpower window functoins for a given set of datapoints.
        All datapoints must share the same bandpower window.

        Parameters
        ----------
        indices: array
            indices of the data points you want windows for

        Returns
        -------
        windows: BandpowerWindow object containing the bandpower window
            functions for these indices.
        """
        ws = unique_list(self.data[i].tags.get('window') for i in indices)
        if len(ws) != 1:
            raise ValueError("You have asked for window functions, "
                             "however, the points you have selected "
                             "have different windows associated to them."
                             "Please narrow down your selection (specify "
                             "tracers and data type) or get windows "
                             "later.")
        ws = ws[0]
        if not isinstance(ws, BandpowerWindow):
            warnings.warn("No bandpower windows associated with these data")
            return None
        w_inds = np.array(self._get_tags_by_index(['window_ind'],indices)[0])
        return ws.get_section(w_inds)

    def get_ell_cl(self, data_type, tracer1, tracer2,
                   return_cov=False, return_ind=False):
        """
        Helper method to extract the ell and C_ell values for a specific
        data type (e.g. 'shear_ee' and pair of tomographic bins)

        Parameters
        ----------
        data_type: str
            Which C_ell type to extract

        tracer1: str
            The name of the first tracer, for example a tomographic bin name

        tracer2: str
            The name of the second tracer

        return_cov: bool
            If True, also return the block of the covariance
            corresponding to these points.  Default=False

        return_ind: bool
            If True, also return the datapoint indices. Default=False

        Returns
        -------
        ell: array
            Ell values for this tracer pair
        mu: array
            Mean values for this tracer pair
        cov_block: 2D array
            (Only if return_cov=True) The block of the covariance for
            these points
        indices: array
            (Only if return_ind=True) datapoint indices.
        """
        return self._get_2pt(data_type, tracer1, tracer2, return_cov,
                             'ell', return_ind)

    def get_theta_xi(self, data_type, tracer1, tracer2,
                     return_cov=False, return_ind=False):
        """
        Helper method to extract the theta and correlation function
        values for a specific data type (e.g. 'shear_xi' and pair of
        tomographic bins).

        Parameters
        ----------

        data_type: str
            Which type of xi to extract

        tracer1: str
            The name of the first tracer, for example a tomographic bin name

        tracer2: str
            The name of the second tracer

        return_cov: bool
            If True, also return the block of the covariance
            corresponding to these points.  Default=False

        return_ind: bool
            If True, also return the datapoint indices. Default=False

        Returns
        -------
        ell: array
            Ell values for this tracer pair

        mu: array
            Mean values for this tracer pair

        cov_block: 2D array
            (Only if return_cov=True) The block of the covariance for
            these points
        indices: array
            (Only if return_ind=True) datapoint indices.
        """
        return self._get_2pt(data_type, tracer1, tracer2, return_cov,
                             'theta', return_ind)

    def _add_2pt(self, data_type, tracer1, tracer2, x, tag_val, tag_name,
                 window, tracers_later, tag_extra=None, tag_extra_name=None):
        """
        Internal method for adding 2pt data points.
        Copes with multiple values for the parameters
        """
        # single data point case
        if np.isscalar(tag_val):
            t = {tag_name: float(tag_val)}
            if tag_extra_name is not None:
                t[tag_extra_name] = tag_extra
            if window is not None:
                t['window'] = window
            self.add_data_point(data_type, (tracer1, tracer2), x,
                                tracers_later=tracers_later, **t)
            return
        # multiple ell/theta values but same bin
        if np.isscalar(tracer1):
            n1 = len(x)
            n2 = len(tag_val)
            if tag_extra_name is None:
                tag_extra = np.zeros(n1)
                n3 = n1
            else:
                n3 = len(tag_extra)
            if not n1 == n2 == n3:
                raise ValueError("Length of inputs do not match in"
                                 f"added 2pt data ({n1},{n2},{n3})")
            if window is None:
                for tag_i, x_i, te_i in zip(tag_val, x, tag_extra):
                    self._add_2pt(data_type, tracer1, tracer2, x_i,
                                  tag_i, tag_name, window,
                                  tracers_later, te_i, tag_extra_name)
            else:
                for tag_i, x_i, w_i, te_i in zip(tag_val, x,
                                                 window, tag_extra):
                    self._add_2pt(data_type, tracer1, tracer2, x_i,
                                  tag_i, tag_name, w_i,
                                  tracers_later, te_i, tag_extra_name)
        # multiple bin values
        elif np.isscalar(data_type):
            n1 = len(x)
            n2 = len(tag_val)
            n3 = len(tracer1)
            n4 = len(tracer2)
            if tag_extra_name is None:
                tag_extra = np.zeros(n1)
                n5 = n1
            else:
                n5 = len(tag_extra)
            if not n1 == n2 == n3 == n4 == n5:
                raise ValueError("Length of inputs do not match in "
                                 f"added 2pt data ({n1},{n2},{n3},{n4},{n5})")
            if window is None:
                for b1, b2, tag_i, x_i, te_i in zip(tracer1, tracer2, tag_val,
                                                    x, tag_extra):
                    self._add_2pt(data_type, b1, b2, x_i, tag_i, tag_name,
                                  window, tracers_later, te_i, tag_extra_name)
            else:
                for b1, b2, tag_i, x_i, w_i, te_i in zip(tracer1,
                                                         tracer2,
                                                         tag_val,
                                                         x,
                                                         window,
                                                         tag_extra):
                    self._add_2pt(data_type, b1, x_i, tag_i, tag_name,
                                  w_i, tracers_later, te_i, tag_extra_name)
        # multiple data point values
        else:
            n1 = len(x)
            n2 = len(tag_val)
            n3 = len(tracer1)
            n4 = len(tracer2)
            n5 = len(data_type)
            if tag_extra_name is None:
                tag_extra = np.zeros(n1)
                n6 = n1
            else:
                n6 = len(tag_extra)
            if not n1 == n2 == n3 == n4 == n5 == n6:
                raise ValueError("Length of inputs do not match in added "
                                 f"2pt data ({n1},{n2},{n3},{n4},{n5},{n6})")
            if window is None:
                for d, b1, b2, tag_i, x_i, te_i in zip(data_type,
                                                       tracer1,
                                                       tracer2,
                                                       tag_val,
                                                       x,
                                                       tag_extra):
                    self._add_2pt(d, b1, b2, x_i, tag_i, tag_name,
                                  window, tracers_later,
                                  te_i, tag_extra_name)
            else:
                for d, b1, b2, tag_i, x_i, w_i, te_i in zip(data_type,
                                                            tracer1,
                                                            tracer2,
                                                            tag_val,
                                                            x,
                                                            window,
                                                            tag_extra):
                    self._add_2pt(d, b1, b2, x_i, tag_i, tag_name,
                                  w_i, tracers_later,
                                  te_i, tag_extra_name)

    def add_ell_cl(self, data_type, tracer1, tracer2, ell, x,
                   window=None, tracers_later=False):
        """
        Add a series of 2pt Fourier space data points, either
        individually or as a group.

        Parameters
        ----------
        data_type: str or array/list of str
            Which type C_ell to add

        tracer1: str or array/list of str
            The name(s) of the first tracer, for example a tomographic bin name

        tracer2: str or array/list of str
            The name(s) of the second tracer

        ell: int or array/list of int/float
            The ell values for these data points

        x: float or array/list of float
            The C_ell values for these data points

        window: Window instance
            Optional window object describing the window function
            of the data point.

        tracers_later: bool
            Optional.  If False (the default), complain if n(z) tracers have
            not yet been defined. Otherwise, suppress this warning

        Returns
        -------
        None

        """
        if isinstance(window, BandpowerWindow):
            if len(ell) != window.nv:
                raise ValueError("Input bandpowers are misshapen")
            tag_extra = range(window.nv)
            tag_extra_name = "window_ind"
            window_use = [window for _ in range(window.nv)]
        else:
            tag_extra = None
            tag_extra_name = None
            window_use = window

        self._add_2pt(data_type, tracer1, tracer2, x, ell, 'ell',
                      window_use, tracers_later,
                      tag_extra, tag_extra_name)

    def add_theta_xi(self, data_type, tracer1, tracer2, theta, x,
                     window=None, tracers_later=False):
        """
        Add a series of 2pt real space data points, either
        individually or as a group.

        Parameters
        ----------
        data_type: str or array/list of str
            Which xi type to extract

        tracer1: str or array/list of str
            The name(s) of the first tracer, for example a tomographic bin name

        tracer2: str or array/list of str
            The name(s) of the second tracer

        theta: float or array/list of int
            The ell values for these data points

        x: float or array/list of float
            The C_ell values for these data points

        window: Window instance
            Optional window object describing the window function
            of the data point.

        tracers_later: bool
            Optional.  If False (the default), complain if n(z) tracers have
            not yet been defined. Otherwise, suppress this warning

        Returns
        -------
        None

        """
        self._add_2pt(data_type, tracer1, tracer2, x, theta, 'theta',
                      window, tracers_later)


def concatenate_data_sets(*data_sets, labels=None, same_tracers=None):
    """Combine multiple sacc data sets together into one.

    In case of two tracers or metadata items with the same name,
    you can use the labels option to pass in a list of strings to append
    to all the names.

    The Covariance will be combined into either a BlockDiagonal covariance or
    a Diagonal covariance, depending on the inputs.  Either all inputs should
    have a covariance attached or none of them.

    Parameters
    ----------
    *data_sets: Sacc objects
        The data sets to combined

    labels: List[str]
        Optional list of strings to append to tracer and metadata names, in
        case of a clash.

    same_tracers: List[str]
        Optional list of tracers that are assumed to be the same in the
        different data_sets but with no correlation between the data points
        involving them. Only the first occurance of each tracer will be added
        to the combined data set.

    Returns
    -------
    output: Sacc object
        The combined data set.

    """
    # Early return of an empty data set object
    if len(data_sets) == 0:
        return Sacc()

    # check for wrong number of labels
    if labels is not None:
        if len(labels) != len(data_sets):
            raise ValueError("Wrong number of labels supplied when "
                             "concatenating data sets")

    # Make same_tracers an empty list for easy comparison
    if same_tracers is None:
        same_tracers = []

    data_0 = data_sets[0]

    # Either all the data sets should have covariances or none of
    # them should.  Concatenating covariances should be
    # straightforward and should always result in a block-diagonal
    # covariance
    if data_0.has_covariance():
        if not all(data_set.has_covariance()
                   for data_set in data_sets):
            raise ValueError("Either all concatenated data sets must "
                             "have covariances, or none of them")
    else:
        if any(data_set.has_covariance()
               for data_set in data_sets):
            raise ValueError("Either all concatenated data sets must "
                             "have covariances, or none of them")

    output = Sacc()

    # Copy the tracers to the new
    for i, data_set in enumerate(data_sets):
        for tracer in data_set.tracers.values():

            # We will be modifying the tracer, so we copy it.
            tracer = copy.deepcopy(tracer)

            # Optionally add a suffix label to avoid name clashes.
            if (labels is not None) and (tracer.name not in same_tracers):
                tracer.name = f'{tracer.name}_{labels[i]}'

            # Check for duplicate tracer names.
            # Probably this happens because the user has not provided
            # any labels to use as tracer suffices.  But it could also
            # happen if they have chosen really really bad labels
            if tracer.name in output.tracers:
                if tracer.name in same_tracers:
                    pass
                elif labels is None:
                    raise ValueError("There is a name clash between "
                                     "tracers in the data sets. "
                                     "Use the labels option to give "
                                     "new names to them")
                else:
                    raise ValueError("After applying your labels "
                                     "there is still a name clash "
                                     "between tracers in your concatenation."
                                     " Try different labels?")
            else:
                # Build up the combined tracer collection
                output.add_tracer_object(tracer)

        for d in data_set.data:
            # Shallow copy because we do not want to clone Window functions,
            # since they are often shared. The reason we do it at all
            # is because we may be modifying the tracers names below.
            d = copy.copy(d)

            # Rename the tracers if required.
            if labels is not None:
                label = labels[i]
                d.tracers = tuple([f'{t}_{label}' for t in d.tracers])
                # Data points might reasonably have a label already,
                # but we would like to add a label from this concatenation
                # process too.  If they have both, we concatenat them.
                # For consistency with the tracers we don't include an
                # underscore
                orig_label = d.get_tag('label', '')
                d.tags['label'] = (f'{orig_label}_{label}'
                                   if orig_label else label)

            # And build up the combined data vector
            output.data.append(d)

    # Combine the covariances
    if data_sets[0].has_covariance():
        covs = [d.covariance for d in data_sets]
        cov = concatenate_covariances(*covs)
        output.add_covariance(cov)

    # Now just the metadata left.
    # It is an error if there is a key that is the same in both
    for i, data_set in enumerate(data_sets):
        for key, val in data_set.metadata.items():

            # Use the label as a suffix here also.
            if labels is not None:
                key = key + labels[i]

            # Check for clashing metadata
            if key in output.metadata:
                raise ValueError("Metadata in concatenated Saccs have "
                                 "same name. Set the labels parameter "
                                 "to fix this.")
            output.metadata[key] = val

    return output




def fix_data_ordering(data_points):
    """
    SACC data points have an ordering column called 'sacc_ordering'
    which is used to keep the data points in the same order as
    the covariance matrix. This function re-orders the data points
    accordingly

    Parameters
    ----------
    data_points: list of DataPoint objects

    Returns
    -------
    ordered_data_points: list of DataPoint objects

    """
    # Older versions of SACC did not have this column, so we
    # check for that situation and if not then add it here, in the
    # order the data points were found in the file.
    # In the old sacc version this order automatically matched the
    # covariance matrix.
    have_ordering = ['sacc_ordering' in dp.tags for dp in data_points]
    if not all(have_ordering):

        if any(have_ordering):
            raise ValueError(
                "Some data points have sacc ordering and some do not. "
                "Hybrid old/new version. This is very wrong. "
                "Please check your data files or ask on #desc-sacc for help."
            )

        print("Warning: The FITS format without the 'sacc_ordering' column is deprecated")
        print("Assuming data rows are in the correct order as it was before version 1.0.")
        for i, dp in enumerate(data_points):
            dp.tags['sacc_ordering'] = i



    # In either case, we now have the 'sacc_ordering' column,
    # so can re-order the data points.
    ordered_data_points = [None for i in range(len(data_points))]
    for dp in data_points:
        i = dp.tags['sacc_ordering']
        ordered_data_points[i] = dp

        # We remove the ordering tag now, as it is not needed
        # in the main library
        del dp.tags['sacc_ordering']

    return ordered_data_points
