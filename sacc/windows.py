import numpy as np
from astropy.table import Table


class BaseWindow:
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
    _window_classes = {}

    def __init_subclass__(cls, window_type):
        # This gets called whenever a subclass is defined.
        # The window_type argument is specified next to the
        # base class in the subclass definition, e.g.
        # window_typ='TopHat', as shown below
        cls._window_classes[window_type] = cls
        cls.window_type = window_type

    @classmethod
    def to_tables(cls, instance_list):
        """Convert a list of BaseWindos to a list of tables.

        This is called when saving data to file.

        The input instances can be different subclasses, and no
        ordering is maintained.

        Parameters
        ----------
        instance_list: list
            List of BaseWindow subclass instances

        Returns
        -------
        table: list
            List of astropy.table.Table instances
        """
        tables = []
        for name, subcls in cls._window_classes.items():
            # Pull out the relevant objects for this subclass.
            # Note that we can't use isinstance here.
            windows = [w for w in instance_list if type(w) == subcls]
            tables += subcls.to_tables(windows)
        return tables

    @classmethod
    def from_tables(cls, table_list):
        """Turn a list of astropy tables into window objects

        This is called when loading data from file.

        Parameters
        ----------
        instance_list: list
            List of BaseWindow instances

        Returns
        -------
        windows: dict
            Dictionary of id -> Window instances
        """
        windows = {}
        for table in table_list:
            subclass_name = table.meta['SACCCLSS']
            subclass = cls._window_classes[subclass_name]
            # Different subclasses can handle this differently.
            windows.update(subclass.from_table(table))
        return windows

    def get_section(self, indices):
        """Return the part of this window corresponding to
        the input indices

        Parameters
        ----------
        indices: array_like
            List of integer indices.

        Returns
        -------
        W: Window object
            Part of the window function corresponding to the
            input indices
        """
        raise NotImplementedError("get_section is not implemented for this "
                                  " window function class.")


class TopHatWindow(BaseWindow, window_type='TopHat'):
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
    def __init__(self, range_min, range_max):
        """Create a top-hat window

        Parameters
        ----------
        range_min: array_like
            The minimum value where the top-hat function equals 1

        range_max: array_like
            The maximum value where the top-hat function equals 1

        """
        self.min = range_min
        self.max = range_max

    @classmethod
    def to_tables(cls, window_list):
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
        tables = []
        for w in window_list:
            cols = [w.min, w.max]
            names = ['min', 'max']
            t = Table(data=cols, names=names)
            t.meta['SACCTYPE'] = 'window'
            t.meta['SACCCLSS'] = cls.window_type
            t.meta['SACCNAME'] = id(w)
            t.meta['EXTNAME'] = 'window:' + cls.window_type
            tables.append(t)
        return tables

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
        return {table.meta['SACCNAME']: cls(table['min'], table['max'])}

    def get_section(self, indices):
        return TopHatWindow(self.min[indices], self.max[indices])


class LogTopHatWindow(TopHatWindow, window_type='LogTopHat'):
    """A window function that is log-constant between two values.

    This object is the same as the TopHat form, except that in between
    the min and max values it is assumed to be constant in the log of the
    argument.  The difference arises when this object is used elsewhere.
    """
    def get_section(self, indices):
        return LogTopHatWindow(self.min[indices], self.max[indices])


class Window(BaseWindow, window_type='Standard'):
    """The Window class defines a tabulated window function.

    The class contains tabulated values of the abscissa (e.g. ell or theta) and
    corresponding weights values for each one. More than one set of weights can
    be used.

    The function could be integrated or summed depending on the
    context.

    Parameters
    ----------
    values: array
        An array of dimension (N) containing the points at which the
        weights are defined.
    weight:
        An array of dimensions (N, N_weight) containing N_weight sets of
        weights corresponding to each value.

    """
    def __init__(self, values, weight):
        self.values = np.array(values)
        self.weight = np.array(weight)

    @classmethod
    def to_tables(cls, window_list):
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
        tables = []
        for w in window_list:
            cols = [w.values, w.weight]
            names = ['values', 'weight']
            t = Table(data=cols, names=names)
            t.meta['SACCTYPE'] = 'window'
            t.meta['SACCCLSS'] = cls.window_type
            t.meta['SACCNAME'] = id(w)
            t.meta['EXTNAME'] = 'window:' + cls.window_type
            tables.append(t)
        return tables

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
        return {table.meta['SACCNAME']: cls(table['values'], table['weight'])}

    def get_section(self, indices):
        return Window(self.values, self.weight[:, indices])
