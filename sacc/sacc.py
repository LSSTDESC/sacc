import copy
import warnings
import os
from io import BytesIO

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .tracers import BaseTracer
from .windows import BaseWindow, BandpowerWindow
from .covariance import BaseCovariance, concatenate_covariances
from .utils import unique_list
from .data_types import standard_types, DataPoint


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

    def __len__(self):
        """
        Return the number of data points in the data set.

        Returns
        -------
        n: int
            The number of data points
        """
        return len(self.data)

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
            elif 'theta' in row.tags:
                return (dt, row.tracers, row.tags['theta'])
            else:
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

    def add_tracer(self, tracer_type, name,
                   *args, **kwargs):
        """
        Add a new tracer

        Parameters
        ----------
        tracer_type: str
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
        tracer = BaseTracer.make(tracer_type, name,
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

    def add_covariance(self, covariance):
        """
        Once you have finished adding data points, add a covariance
        for the entire set.

        Parameters
        ----------
        covariance: array or list
            2x2 numpy array containing the covariance of the added data points
            OR a list of blocks

        Returns
        -------
        None
        """
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
        if indices.dtype != np.bool:
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
        if indices.dtype != np.bool:
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
            if not ((data_type is None or d.data_type == data_type)):
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

    def get_data_types(self):
        """
        Get a list of the different data types stored in the Sacc

        Returns
        --------
        data_types: list of strings
            A list of the string data types in the data set
        """
        return unique_list(d.data_type for d in self.data)

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
        all_windows = unique_list(d.get_tag('window') for d in self.data)
        window_ids = {w: id(w) for w in all_windows}
        tables = BaseWindow.to_tables(all_windows)
        return tables, window_ids

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

        # Since we don't want to re-order the file as a side effect
        # we first make a copy of ourself and re-order that.
        S = self.copy()
        S.to_canonical_order()

        # Tables for the windows
        tables, window_ids = S._make_window_tables()
        lookup = {'window': window_ids}

        # Tables for the tracers
        tables += BaseTracer.to_tables(S.tracers.values())

        # Tables for the data sets
        for dt in S.get_data_types():
            data = S.get_data_points(dt)
            table = DataPoint.to_table(data, lookup)
            # Could move this inside to_table?
            table.meta['SACCTYPE'] = 'data'
            table.meta['SACCNAME'] = dt
            table.meta['EXTNAME'] = f'data:{dt}'
            tables.append(table)

        # Create the actual fits object
        hdr = fits.Header()

        # save any global metadata in the header.
        # We save the keys and values as separate header cards,
        # because otherwise the keys are all forced to upper case
        hdr['NMETA'] = len(S.metadata)
        for i, (k, v) in enumerate(S.metadata.items()):
            hdr[f'KEY{i}'] = k
            hdr[f'VAL{i}'] = v
        hdus = [fits.PrimaryHDU(header=hdr)] + \
               [fits.table_to_hdu(table) for table in tables]

        # Covariance, if needed.
        # All the other data elements become astropy tables first,
        # But covariances are a bit more complicated and dense, so we
        # allow them to convert straight to
        if S.covariance is not None:
            hdus.append(S.covariance.to_hdu())

        # Make and save the final FITS data
        hdu_list = fits.HDUList(hdus)

        # The astropy writeto shows very poor performance
        # when writing lots of small metadata strings on
        # the NERSC Lustre file system.  So we write to
        # a buffer first and then save that.

        # First we have to manually check for overwritten files
        # We raise the same error as astropy
        if os.path.exists(filename) and not overwrite:
            raise OSError(f"File {filename} already exists and overwrite=False")

        # Create the buffer and write the data to it
        buf = BytesIO()
        hdu_list.writeto(buf)

        # Rewind and read the binary data we just wrote
        buf.seek(0)
        output_data = buf.read()

        # Write the binary data to the target file
        with open(filename, "wb") as f:
            f.write(output_data)

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
        hdu_list = fits.open(filename, "readonly")

        # Split the HDU's into the different sacc types
        tracer_tables = [Table.read(hdu)
                         for hdu in hdu_list
                         if hdu.header.get('SACCTYPE') == 'tracer']
        window_tables = [Table.read(hdu)
                         for hdu in hdu_list
                         if hdu.header.get('SACCTYPE') == 'window']
        data_tables = [Table.read(hdu) for hdu in hdu_list
                       if hdu.header.get('SACCTYPE') == 'data']
        cov = [hdu for hdu in hdu_list if hdu.header.get('SACCTYPE') == 'cov']

        # Pull out the classes for these components.
        tracers = BaseTracer.from_tables(tracer_tables)
        windows = BaseWindow.from_tables(window_tables)

        # The lookup table is used to convert from ID numbers to
        # Window objects.
        lookup = {'window': windows}

        # Collect together all the data points from the different sections
        data = []
        for table in data_tables:
            data += DataPoint.from_table(table, lookup)

        # Finally, take all the pieces that we have collected
        # and add them all into this data set.
        S = cls()
        for tracer in tracers.values():
            S.add_tracer_object(tracer)

        # Add the data points manually instead of using the API, since we
        # have already constructed them.
        for d in data:
            S.data.append(d)

        # Assume there is only a single covariance extension,
        # if there are any
        if cov:
            S.add_covariance(BaseCovariance.from_hdu(cov[0]))

        # Load metadata from the primary heaer
        header = hdu_list[0].header

        # Load each key,value pair in turn.
        # This will work for normal scalar data types;
        # arrays etc. will need some thought.
        n_meta = header['NMETA']
        for i in range(n_meta):
            k = header[f'KEY{i}']
            v = header[f'VAL{i}']
            S.metadata[k] = v

        hdu_list.close()

        return S

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
            else:
                return angle, mu, cov_block
        else:
            if return_ind:
                return angle, mu, ind
            else:
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
            warnings.warn("No bandpower windows associated to these data")
            return None
        else:
            w_inds = np.array(self._get_tags_by_index(['window_ind'],
                                                      indices)[0])
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
        elif np.isscalar(tracer1):
            n1 = len(x)
            n2 = len(tag_val)
            if tag_extra_name is None:
                tag_extra = np.zeros(n1)
                n3 = n1
            else:
                n3 = len(tag_extra)
            if not (n1 == n2 == n3):
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
            if not (n1 == n2 == n3 == n4 == n5):
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
            if not (n1 == n2 == n3 == n4 == n5 == n6):
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
            window_use = [window for i in range(window.nv)]
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


def concatenate_data_sets(*data_sets, labels=None):
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
            if labels is not None:
                tracer.name = f'{tracer.name}_{labels[i]}'

            # Check for duplicate tracer names.
            # Probably this happens because the user has not provided
            # any labels to use as tracer suffices.  But it could also
            # happen if they have chosen really really bad labels
            if tracer.name in output.tracers:
                if labels is None:
                    raise ValueError("There is a name clash between "
                                     "tracers in the data sets. "
                                     "Use the labels option to give "
                                     "new names to them")
                else:
                    raise ValueError("After applying your labels "
                                     "there is still a name clash "
                                     "between tracers in your concatenation."
                                     " Try different labels?")
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
