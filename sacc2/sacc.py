import numpy as np
import pickle
import copy
import os

from .tracers import Tracer

# add more as we develop them
allowed_types = [
    "shear_xi_plus",
    "shear_xi_minus",
    "shear_ee",
    "shear_bb",
    "galaxy_density_cl",
    "galaxy_density_w",
    "ggl_gamma_t",
    "ggl_gamma_x",
    "ggl_E",
    "ggl_B",
]



class DataPoint:
    def __init__(self, data_type, tracers, value, **tags):
        self.data_type = data_type
        self.tracers = tracers
        self.value = value
        self.tags = tags
    
    def __repr__(self):
        return f"<Data {self.data_type} {self.tracers} {self.value} {self.tags}>s"
    
    def get_tag(self, tag):
        return self.tags.get(tag)

    def __getitem__(self, item):
        return self.tags[item]


        
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

    def __len__(self):
        """
        Return the number of data points in the data set.
        """
        return len(self.data)

    def copy(self):
        """
        Create a copy of the data set with no data shared with the original
        """
        return copy.deepcopy(self)

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

        T = Tracer.make(tracer_type, name, *args, *kwargs)
        self.tracers[name] = T

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
        covariance: array
            2x2 numpy array containing the covariance of the added data points

        Returns
        -------
        None
        """
        # first deal with 1-dimensional data
        covariance = np.atleast_2d(covariance)
        # check that the covariance is the same size as the data
        n = len(self)
        if not covariance.shape == (n,n):
            raise ValueError(f"Covariance has wrong size or shape "
                f"{covariance.shape} for number of data points {n}")
        
        # everything is fine so just set the result
        self.covariance = covariance


    def cut(self, mask):
        """
        Remove data points and corresponding covariance elements following a mask, in-place.

        You can find indices (e.g. matching some tag) to remove using the indices method.

        Parameters
        ----------
        mask: array or list
            Mask must be either a boolean array or a list/array of integer indices to remove.
            if boolean then True means to cut data point and False means to keep it
            if indices then values indicate data points to cut out
        """
        mask = np.array(mask)
            
        if mask.dtype == np.bool:
            if not len(mask)==len(self):
                raise ValueError("Mask passed in is wrong size")
            self.data = [d for i,d in enumerate(self.data) if not mask[i]]
        else:
            # slow
            self.data = [d for i,d in enumerate(self.data) if not i in mask]
        
        self.covariance = self.covariance[mask][:,mask]

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
        for i,d in enumerate(self.data):
            # Skip things with the wrong type or tracer
            if not ((tracers is None) or (d.tracers == tracers)):
                continue
            if not ((data_type is None or d.data_type == data_type)):
                continue
            # Remove any objects that don't match the required tags,
            # including the fact that we can specify tag__lt and tag__gt
            # in order to remove/accept ranges
            ok = True
            for name,val in select.items():
                if name.endswith("__lt"):
                    name = name[:-4]
                    if not d.get_tag(name) < val:
                        ok=False
                        break
                elif name.endswith("__gt"):
                    name = name[:-4]
                    if not d.get_tag(name) > val:
                        ok=False
                        break
                else:
                    if not d.get_tag(name) == val:
                        ok=False
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
        values = [[d.get_tag(tag) for i,d in enumerate(self.data) if i in indices]
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
        indices = set(self.indices(data_type=data_type, tracers=tracers, **select))
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
        s = {d.data_type for d in self.data}
        return list(s)

    def get_tracer(self, name):
        """
        Get the tracer object with the given name

        Parameters
        -----------
        name: str
            A string name of a tracer

        Returns
        -------
        tracer: Tracer object
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
        return list(set([self.data[i].tracers for i in indices]))
        

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
                " but data is length {}".format(len(mu),len(self.data)))
        for m, d in zip(mu, self.data):
            d.value = m


    def save(self, filename, overwrite=False):
        """
        Save the data set to a file using the pickle format

        Parameters
        ----------
        filename: str
            The file to save to

        overwrite: bool
            If True, overwrite the file if it exists already.
            Otherwise, raises OSError.

        """
        if os.path.exists(filename) and not overwrite:
            raise OSError(f"File {filename} already exists. Set overwrite=True to replace it.")
        with open(filename, "wb") as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, filename):
        """
        Load a dataset from the pickle format.

        Parameters
        ----------
        filename: str
            The file to load from
        """
        with open(filename, "rb") as f:
            S = pickle.load(f)
        return S

    #
    # Methods below here are helper functions for specific types of data.
    # We can add more of them as it becomes clear what people need.
    # 
    #

    def _get_2pt(self, data_type, bin1, bin2, return_cov, angle_name):
        # Internal helper method for get_ell_cl and get_theta_xi
        ind = self.indices(data_type, (bin1,bin2))

        mu = np.array(self.mean[ind])
        angle = np.array(self._get_tags_by_index([angle_name], ind)[0])

        if return_cov:
            if self.covariance is None:
                raise ValueError("This sacc data does not have a covariance attached")
            cov_block = self.covariance[ind][:,ind]
            angle, mu, cov_block
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
            t = {tag_name:tag_val}
            if window is not None:
                t['window'] = window
            self.add_data_point(data_type, (bin1,bin2), x, **t)
            return
        # multiple ell/theta values but same bin
        elif np.isscalar(bin1):
            n1 = len(x)
            n2 = len(tag_val)
            if not n1==n2:
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
                for b1, b2, tag_i, x_i in zip(bin1, bin2, tag, x):
                    self._add_2pt(data_type, b1, b2, x_i, tag_i, tag_name, window)
            else:
                for b1, b2, tag_i, x_i, w_i in zip(bin1, bin2, tag, x, window):
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
                for d, b1, b2, tag_i, x_i in zip(data_type, bin1, bin2, tag, x):
                    self._add_2pt(d, b1, b2, x_i, tag_i, tag_name, window)
            else:
                for d, b1, b2, tag_i, x_i, w_i in zip(data_type, bin1, bin2, tag, x, window):
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
        
