import numpy as np
import copy
import warnings

from astropy.io import fits
from astropy.table import Table, Column

from .tracers import BaseTracer
from .windows import BaseWindow
from .covariance import BaseCovariance
from .utils import unique_list
from .data_types import known_types, known_types_list

# These null values are used in place
# of missing values.
null_values = {
    'i': -1437530437530211245,
    'f': -1.4375304375e30,
    'U': '',
}


def hide_null_values(table):
    for name, col in list(table.columns.items()):
        if col.dtype.kind == 'O':
            good_values = [x for x in col if x is not None]
            good_kind = np.array(good_values).dtype.kind
            null = null_values[good_kind]
            good_col = np.array([null if x is None else x for x in col])
            table[name] = Column(good_col)


class DataPoint:
    def __init__(self, data_type, tracers, value, **tags):
        self.data_type = data_type
        self.tracers = tracers
        self.value = value
        self.tags = tags
        if data_type not in known_types:
            warnings.warn(f"Unknown data_type value {data_type}. If possible use a pre-defined type, or add to the list.")

    def __repr__(self):
        return f"<Data {self.data_type} {self.tracers} {self.value} {self.tags}>s"

    def get_tag(self, tag):
        return self.tags.get(tag)

    def __getitem__(self, item):
        return self.tags[item]

    @staticmethod
    def _choose_fields(data):
        tags = set()
        ntracer = 0
        for d in data:
            ntracer = max(ntracer, len(d.tracers))
            tags.update(d.tags.keys())
        tags = list(tags)
        tracers = [f'tracer_{i}' for i in range(ntracer)]
        return tracers, tags

    @classmethod
    def to_table(cls, data, lookups={}):
        """
        Convert a list of data points to a single homogenous table

        Since data points can have varying tags, this method uses
        null values to represent non-present tags.

        Parameters
        ----------

        data: list
            A list of DataPoint objects

        lookups: dict
            A dictionary of tags->dict showing replacements to make
            in the tags. Default is empty.

        Returns

        table: astropy.table.Table
            table object containing data points
        """
        # Get the names of the columns to generate
        tracers, tags = cls._choose_fields(data)
        names = tracers + ['value'] + tags
        ntracer = len(tracers)
        # Convert each data point to a row
        rows = [d._make_row(tracers, tags, lookups) for d in data]

        # Convert to a table and fiddle slightly.
        table = Table(rows=rows, names=names)
        table.meta['NTRACER'] = ntracer
        hide_null_values(table)
        return table

    @classmethod
    def from_table(cls, table, lookups={}):
        """
        Convert a table back into a list of data points.

        This method removes null values from the tags.

        Parameters
        ----------

        table: astropy.table.Table
            A table of data containing the tracers, values, and tags

        lookups: dict
            A dictionary of tags->dict showing replacements to make
            in the tags. Default is empty.

        Returns
        -------

        data: list
            list of DataPoint objects
        """
        # Get out required table metadata
        nt = table.meta['NTRACER']
        data_type = table.meta['SACCNAME']

        # Tag names - we will remove missing tags below
        tag_names = table.colnames[nt + 1:]
        data = []
        for row in table:
            # Get basic data elements
            tracers = tuple([row[f'tracer_{i}'] for i in range(nt)])
            value = row['value']

            # Deal with tags.  First just pull out all remaining columns
            tags = {name: row[name] for name in tag_names}
            for k, v in list(tags.items()):
                # Deal with any tags that we should replace.
                # This is mainly used for Window instances.
                if k in lookups:
                    tags[k] = lookups[k].get(v, v)
                # Now delete and null values, as indicated by the sentinel above.
                if hasattr(tags[k], 'dtype') and v == null_values[tags[k].dtype.kind]:
                    del tags[k]
            # Finally convert back to a data point and record
            data_point = cls(data_type, tracers, value, **tags)
            data.append(data_point)
        return data

    def _make_row(self, tracers, tags, lookups):
        # Turn this data point into a list with specified tracers and tags.
        # If some tracers or tags are missing (homogenous data set) then
        # use blank values or Nones for them.
        nt = len(tracers)
        missing = nt - len(self.tracers)
        row = list(self.tracers) + ["" for i in range(missing)]
        row.append(self.value)
        for t in tags:
            v = self.tags.get(t)
            lookup = lookups.get(t)
            if lookup is not None:
                v = lookup.get(v, v)
            row.append(v)
        return row


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
            if row.data_type in known_types_list:
                dt = known_types_list.index(row.data_type)
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
        indices = [i[0] for i in sorted(enumerate(self.data), key=lambda x:order_key(x[1]))]

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

        if self.covariance is not None:
            self.covariance = self.covariance.masked(indices)

    #
    # Builder methods for building up Sacc data from scratch in memory
    #

    def add_tracer(self, tracer_type, name, *args, **kwargs):
        """
        Add a new tracer
        Find the indices of all points matching the given selection

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

        tracer = BaseTracer.make(tracer_type, name, *args, *kwargs)
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

    def add_data_point(self, data_type, tracers, value, tracers_later=False, **tags):
        """
        Add a data point to the set.

        Parameters
        ----------
        data_type: str

        tracers: tuple of str
            Strings corresponding to zero or more of the tracers in the data set
            These should either be already set up using the add_tracer method,
            or you could set tracers_laster=True if you want to add them later.
            e.g. for 2pt measurements the tracers are the names of the two n(z)
            samples

        value: float
            A single value for the data point

        tracers_later: bool
            If True, do not complain if the tracers are not know already

        **tags:
            Tags to apply to this data point.
            Tags can be any arbitrary metadata that you might want later,
            For 2pt data the tag would include an angle theta or ell.

        Returns
        -------
        None
        """
        if self.covariance is not None:
            raise ValueError("You cannot add a data point after setting the covariance")
        tracers = tuple(tracers)
        for tracer in tracers:
            if (tracer not in self.tracers) and (not tracers_later):
                raise ValueError(f"Tracer named '{tracer}' is not in the known list of tracers."
                                 "Either put it in before adding data points or set tracers_later=True")
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
            self.covariance = covariance
        else:
            self.covariance = BaseCovariance.make(covariance, len(self))

    def _indices_to_bool(self, mask):
        # Convert an array of indices into a boolean True mask
        if mask.dtype not in [np.int8, np.int16, np.int32, np.int64]:
            raise ValueError(f"Wrong mask type ({mask.  dtype}) - expected integers or boolean")
        m = np.zeros(len(self), dtype=bool)
        for i in mask:
            m[i] = True
        return m

    def mask(self, mask):
        """
        Select data points, keeping only values where the mask is True or an index is
        included in it.

        You can use Sacc.cut to do the opposite operation, keeping points where the mask
        is False.

        You can find indices (e.g. matching some tag) to keep using the Sacc.indices method.

        Parameters
        ----------
        mask: array or list
            Mask must be either a boolean array or a list/array of integer indices to remove.
            if boolean then True means to keep a data point and False means to cut it
            if integers then values indicate data points to keep
        """
        mask = np.array(mask)

        # Convert integer masks to booleans
        if mask.dtype != np.bool:
            mask = self._indices_to_bool(mask)

        self.data = [d for i, d in enumerate(self.data) if mask[i]]
        if self.covariance is not None:
            self.covariance = self.covariance.masked(mask)

    def cut(self, cut):
        """
        Remove data points, getting rid of points where the mask is True or an index is
        included in it.

        You can use Sacc.mask to do the opposite operation, keeping points where the mask
        is True.

        You can find indices (e.g. matching some tag) to keep using the Sacc.indices method.

        Parameters
        ----------
        cut: array or list
            Mask must be either a boolean array or a list/array of integer indices to remove.
            if boolean then True means to cut data point and False means to keep it
            if integers then values indicate data points to cut out
        """
        cut = np.array(cut)

        # Convert integer masks to booleans
        if cut.dtype != np.bool:
            cut = self._indices_to_bool(cut)

        # Get the mask method to do the actual work
        self.mask(~cut)

    def indices(self, data_type=None, tracers=None, **select):
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
        return np.array(indices, dtype=int)

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
            For each input tag, a corresponding list of the value of that tag for given
            selection, in the order the matching data points were added.


        """
        indices = set(indices)
        values = [[d.get_tag(tag) for i, d in enumerate(self.data) if i in indices]
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
            For each input tag, a corresponding list of the value of that tag for given
            selection, in the order the matching data points were added.


        """
        indices = self.indices(data_type=data_type, tracers=tracers, **select)
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
        return self.get_tags([tag], data_type=data_type, tracers=tracers, **select)[0]

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
                             " but data is length {}".format(len(mu), len(self.data)))
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
        for k, v in S.metadata.items():
            hdr[k] = v
        hdus = [fits.PrimaryHDU(header=hdr)] + [fits.table_to_hdu(table) for table in tables]

        # Covariance, if needed.
        # All the other data elements become astropy tables first,
        # But covariances are a bit more complicated and dense, so we
        # allow them to convert straight to
        if S.covariance is not None:
            hdus.append(S.covariance.to_hdu())

        # Make and save the final FITS data
        hdu_list = fits.HDUList(hdus)
        hdu_list.writeto(filename, overwrite=overwrite)

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
        hdu_list = fits.open(filename)

        # Split the HDU's into the different sacc types
        tracer_tables = [Table.read(hdu) for hdu in hdu_list if hdu.header.get('SACCTYPE') == 'tracer']
        window_tables = [Table.read(hdu) for hdu in hdu_list if hdu.header.get('SACCTYPE') == 'window']
        data_tables = [Table.read(hdu) for hdu in hdu_list if hdu.header.get('SACCTYPE') == 'data']
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

        return S

    #
    # Methods below here are helper functions for specific types of data.
    # We can add more of them as it becomes clear what people need.
    #
    #

    def _get_2pt(self, data_type, bin1, bin2, return_cov, angle_name):
        # Internal helper method for get_ell_cl and get_theta_xi
        ind = self.indices(data_type, (bin1, bin2))

        mu = np.array(self.mean[ind])
        angle = np.array(self._get_tags_by_index([angle_name], ind)[0])

        if return_cov:
            if self.covariance is None:
                raise ValueError("This sacc data does not have a covariance attached")
            cov_block = self.covariance.get_block(ind)
            return angle, mu, cov_block
        else:
            return angle, mu

    def get_ell_cl(self, data_type, bin1, bin2, return_cov=False):
        """
        Helper method to extract the ell and C_ell values for a specific
        data type (e.g. 'shear_ee' and pair of tomographic bins)

        Parameters
        ----------

        data_type: str
            Which C_ell to extract
        bin1: str
            The name of the first tomographic bin
        bin2: str
            The name of the second tomographic bin
        return_cov: bool
            If True, also return the block of the covariance
            corresponding to these points.  Default=False

        Returns:
        ell: array
            Ell values for this bin pair
        mu: array
            Mean values for this bin pair
        cov_block: 2x2 array
            (Only if return_cov=True) The block of the covariance for
            these points
        """
        return self._get_2pt(data_type, bin1, bin2, return_cov, 'ell')

    def get_theta_xi(self, data_type, bin1, bin2, return_cov=False):
        """
        Helper method to extract the theta and correlation function values for a specific
        data type (e.g. 'shear_xi' and pair of tomographic bins)

        Parameters
        ----------

        data_type: str
            Which C_ell to extract
        bin1: str
            The name of the first tomographic bin
        bin2: str
            The name of the second tomographic bin
        return_cov: bool
            If True, also return the block of the covariance
            corresponding to these points.  Default=False

        Returns:
        ell: array
            Ell values for this bin pair
        mu: array
            Mean values for this bin pair
        cov_block: 2x2 array
            (Only if return_cov=True) The block of the covariance for
            these points
        """
        return self._get_2pt(data_type, bin1, bin2, return_cov, 'theta')

    def _add_2pt(self, data_type, bin1, bin2, x, tag_val, tag_name, window):
        """
        Internal method for adding 2pt data points.
        Copes with multiple values for the parameters
        """
        # single data point case
        if np.isscalar(tag_val):
            t = {tag_name: float(tag_val)}
            if window is not None:
                t['window'] = window
            self.add_data_point(data_type, (bin1, bin2), x, **t)
            return
        # multiple ell/theta values but same bin
        elif np.isscalar(bin1):
            n1 = len(x)
            n2 = len(tag_val)
            if not n1 == n2:
                raise ValueError(f"Length of inputs do not match in added 2pt data ({n1},{n2})")
            if window is None:
                for tag_i, x_i in zip(tag_val, x):
                    self._add_2pt(data_type, bin1, bin2, x_i, tag_i, tag_name, window)
            else:
                for tag_i, x_i, w_i in zip(tag_val, x, window):
                    self._add_2pt(data_type, bin1, bin2, x_i, tag_i, tag_name, w_i)
        # multiple bin values
        elif np.isscalar(data_type):
            n1 = len(x)
            n2 = len(tag_val)
            n3 = len(bin1)
            n4 = len(bin2)
            if not (n1 == n2 == n3 == n4):
                raise ValueError(f"Length of inputs do not match in added 2pt data ({n1}, {n2}, {n3}, {n4})")
            if window is None:
                for b1, b2, tag_i, x_i in zip(bin1, bin2, tag_val, x):
                    self._add_2pt(data_type, b1, b2, x_i, tag_i, tag_name, window)
            else:
                for b1, b2, tag_i, x_i, w_i in zip(bin1, bin2, tag_val, x, window):
                    self._add_2pt(data_type, b1, x_i, tag_i, tag_name, w_i)
        # multiple data point values
        else:
            n1 = len(x)
            n2 = len(tag_val)
            n3 = len(bin1)
            n4 = len(bin2)
            n5 = len(data_type)
            if not (n1 == n2 == n3 == n4 == n5):
                raise ValueError(f"Length of inputs do not match in added 2pt data ({n1}, {n2}, {n3}, {n4}, {n5})")
            if window is None:
                for d, b1, b2, tag_i, x_i in zip(data_type, bin1, bin2, tag_val, x):
                    self._add_2pt(d, b1, b2, x_i, tag_i, tag_name, window)
            else:
                for d, b1, b2, tag_i, x_i, w_i in zip(data_type, bin1, bin2, tag_val, x, window):
                    self._add_2pt(d, b1, b2, x_i, tag_i, tag_name, w_i)

    def add_ell_cl(self, data_type, bin1, bin2, ell, x, window=None):
        """
        Add a series of 2pt Fourier space data points, either
        individually or as a group.

        data_type: str or array/list of str
            Which C_ell to extract
        bin1: str or array/list of str
            The name(s) of the first tomographic bin
        bin2: str or array/list of str
            The name(s) of the second tomographic bin
        ell: int or array/list of int/float
            The ell values for these data points
        x: float or array/list of float
            The C_ell values for these data points
        window: Window instance
            Optional window object describing the window function
            of the data point.


        Returns
        -------
        None

        """
        self._add_2pt(data_type, bin1, bin2, x, ell, 'ell', window)

    def add_theta_xi(self, data_type, bin1, bin2, theta, x, window=None):
        """
        Add a series of 2pt real space data points, either
        individually or as a group.

        data_type: str or array/list of str
            Which C_ell to extract
        bin1: str or array/list of str
            The name(s) of the first tomographic bin
        bin2: str or array/list of str
            The name(s) of the second tomographic bin
        theta: float or array/list of int
            The ell values for these data points
        x: float or array/list of float
            The C_ell values for these data points
        window: Window instance
            Optional window object describing the window function
            of the data point.

        Returns
        -------
        None

        """
        self._add_2pt(data_type, bin1, bin2, x, theta, 'theta', window)