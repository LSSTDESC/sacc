import numpy as np
from astropy.table import Table
from .io import BaseIO, MULTIPLE_OBJECTS_PER_TABLE, ONE_OBJECT_PER_TABLE

class BaseWindow(BaseIO):
    """Base class for window functions.

    Window functions here are for 1D variables and describe
    binnings from functions onto discrete values.

    They are treated as a special kind of tag value in Sacc.

    Subclasses describe common types of window function, including
    a top hat between two values and a general tabulated
    functional form.

    This base class has class methods that can be used to turn
    mixed lists of windows to/from astropy tables, for I/O.
    """
    _sub_classes = {}


class TopHatWindow(BaseWindow, type_name='TopHat'):
    """A window function that is constant between two values.

    The top-hat is zero elsewhere.

    In the case of discrete functions like ell window where it matters, these
    window functions should follow the python convention and
    the upper limit should be exclusive, so that you can use
    range(win.min, win.max) to select the right ell values.

    Parameters
    ----------
    min: int/float
        The minimum value where the top-hat function equals 1

    max: int/float
        The maximum value where the top-hat function equals 1
    """
    storage_type = MULTIPLE_OBJECTS_PER_TABLE
    def __init__(self, range_min, range_max):
        """Create a top-hat window

        Parameters
        ----------
        range_min: int/float
            The minimum value where the top-hat function equals 1

        range_max: int/float
            The maximum value where the top-hat function equals 1

        """
        self.min = range_min
        self.max = range_max

    def __eq__(self, other):
        """Equality test.

        Two TopHatWindows are equal if they have the same min and max.

        Parameters
        ----------
        other: object
            The other object to test for equality
        """
        if not isinstance(other, type(self)):
            return False
        return self.min == other.min and self.max == other.max

    def __hash__(self):
        """Hash function.

        This uses the same attributes as __eq__ to ensure consistent
        behaviour.

        Returns
        -------
        hash: int

        """
        return hash((self.min, self.max))

    @classmethod
    def to_table(cls, window_list):
        """Convert a list of Top-Hat windows to a list of astropy tables.

        A single table is created for all the windows.

        The tables contain an ID column which uniquely identifies the
        window instance, but only whilst running a given python process -
        it is not portable.  It should not be used for anythng other than I/O.

        Parameters
        ----------
        instance_list: list
            List of TopHatWindow instances

        Returns
        -------
        table: list
            List of astropy.table.Table instances
        """
        mins = [w.min for w in window_list]
        maxs = [w.max for w in window_list]
        ids = [id(w) for w in window_list]
        return Table(data=[ids, mins, maxs], names=['id', 'min', 'max'])

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table to a dictionary of Top-Hat windows

        Multiple windows are extracted from the single table.

        Parameters
        ----------
        table: astropy.table.Table

        Returns
        -------
        windows: dict
            Dictionary of id -> TopHatWindow instances
        """
        return {row['id']: cls(row['min'], row['max']) for row in table}


class LogTopHatWindow(TopHatWindow, type_name='LogTopHat'):
    """A window function that is log-constant between two values.

    This object is the same as the TopHat form, except that in between
    the min and max values it is assumed to be constant in the log of the
    argument.  The difference arises when this object is used elsewhere.
    """


class Window(BaseWindow, type_name='Standard'):
    """The Window class defines a tabulated window function.

    The class contains tabulated values of the abscissa (e.g. ell or theta) and
    corresponding weights values for each one.

    The function could be integrated or summed depending on the
    context.

    Parameters
    ----------
    values: array
        The points at which the weights are defines
    weight:
        The weights corresponding to each value

    """
    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, values, weight):
        self.values = np.array(values)
        self.weight = np.array(weight)

    def __eq__(self, other):
        """Equality test

        Two Windows are equal if they have equivalent values and weights.

        Parameters
        ----------
        other: Window
            The other Window to compare

        """
        if not isinstance(other, type(self)):
            return False
        return np.allclose(self.values, other.values) and np.allclose(self.weight, other.weight)

    def __hash__(self):
        """Hash function.

        This uses the identity of the object. Caution: this is not
        ideal, because it means that two windows with equivalent values and
        weights may not have the same hash.

        Returns
        -------
        hash: int
        """
        return id(self)

    def to_table(self):
        """Convert a list of windows to a list of astropy tables.

        One table is created per window.

        The tables contain an ID column which uniquely identifies the
        window instance, but only whilst running a given python process -
        it is not portable.  It should not be used for anythng other than I/O.

        Parameters
        ----------
        instance_list: list
            List of Window instances

        Returns
        -------
        table: list
            List of astropy.table.Table instances
        """
        cols = [self.values, self.weight]
        names = ['values', 'weight']
        t = Table(data=cols, names=names)
        return t

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table to a dictionary of Windows

        One window is extracted from the table.

        Parameters
        ----------
        table: astropy.table.Table

        Returns
        -------
        windows: dict
            Dictionary of id -> Window instances
        """
        return cls(table['values'], table['weight'])


class BandpowerWindow(BaseWindow, type_name='Bandpower'):
    """The BandpowerWindow class defines a tabulated for power
    spectrum bandpowers.

    The class contains tabulated values of ell and corresponding weights
    values for each one. More than one set of weights can be used.

    Parameters
    ----------
    values: array
        An array of dimension (N_ell) containing the ell values at which the
        weights are defined.
    weight:
        An array of dimensions (N_ell, N_weight) containing N_weight sets of
        weights corresponding to each value.

    """
    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, values, weight):
        nl, nv = weight.shape
        nell = len(values)
        if nl != len(values):
            raise ValueError(f"Wrong input shapes ${nl}!=${nell}")
        self.nell = nell
        self.nv = nv
        self.values = np.array(values)
        self.weight = np.array(weight)

    def __eq__(self, other):
        """Equality test

        Two BandpowerWindows are equal if they have the same values and weights.

        Parameters
        ----------
        other: BandpowerWindow
            The other BandpowerWindow to compare

        Returns
        -------
        bool
            True if the windows are equal, False otherwise
        """
        if not isinstance(other, type(self)):
            return False
        return self.nell == other.nell and \
            self.nv == other.nv and \
            np.allclose(self.values, other.values) and \
            np.allclose(self.weight, other.weight)

    def __hash__(self):
        """Hash function.

        This uses the identity of the object. Caution: this is not ideal,
        because it means that two windows with equivalent values and
        weights may not have the same hash.

        Returns
        -------
        hash: int
        """
        return id(self)

    def to_table(self):
        """Convert a list of windows to a list of astropy tables.

        One table is created per window.

        The tables contain an ID column which uniquely identifies the
        window instance, but only whilst running a given python process -
        it is not portable.  It should not be used for anythng other than I/O.

        Parameters
        ----------
        instance_list: list
            List of Window instances

        Returns
        -------
        table: list
            List of astropy.table.Table instances
        """
        cols = [self.values, self.weight]
        names = ['values', 'weight']
        t = Table(data=cols, names=names)
        return t

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table to a dictionary of Windows

        One window is extracted from the table.

        Parameters
        ----------
        table: astropy.table.Table

        Returns
        -------
        windows: dict
            Dictionary of id -> Window instances
        """
        return cls(table['values'], table['weight'])

    def get_section(self, indices):
        """Get part of this window function corresponding to the input
        indices.

        Parameters
        ----------
        indices: int or array_like
            Indices to return.

        Returns
        -------
        window: `Window`
            A `Window object.
        """
        return self.__class__(self.values, self.weight[:, indices])
