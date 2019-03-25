import numpy as np
from astropy.io import fits
from astropy.table import Table

class BaseWindow:
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
        """
        Convert a list of BaseWindow objects, possibly
        of different subclasses, to a list of output
        tables.  No ordering is maintained.

        Parameters
        ----------
        instance_list: list
            List of BaseWindow instances

        Returns
        -------
        table: list
            List of astropy.table.Table instances
        """
        tables = []
        for name, subcls in cls._window_classes.items():
            # Pull out the relevant objects for this subclass.
            # Note that we can't use isinstance here.
            windows = [w for w in instance_list if type(w)==subcls]
            tables += subcls.to_tables(windows)
        return tables


    @classmethod
    def from_tables(cls, table_list):
        """
        Parse a list of tables containing window data into
        a dictionary of different window types

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


class TopHatWindow(BaseWindow, window_type='TopHat'):
    def __init__(self, range_min, range_max):
        self.min = range_min
        self.max = range_max

    @classmethod
    def to_tables(cls, window_list):
        mins = [w.min for w in window_list]
        maxs = [w.max for w in window_list]
        ids  = [id(w) for w in window_list]
        t = Table(data=[ids,mins,maxs], names=['id', 'min', 'max'])
        t.meta['SACCTYPE'] = 'window'
        t.meta['SACCCLSS'] = cls.window_type
        t.meta['EXTNAME'] = 'window:'+cls.window_type
        return [t]

    @classmethod
    def from_table(cls, table):
        return {row['id']: cls(row['min'], row['max']) for row in table}



class Window(BaseWindow, window_type='Standard'):
    def __init__(self, values, weight):
        self.values = np.array(values)
        self.weight = np.array(weight)

    @classmethod
    def to_tables(cls, window_list):
        tables = []
        for w in window_list:
            cols = [w.values, w.weight]
            names = ['values', 'weight']
            t = Table(data=cols, names=names)
            t.meta['SACCTYPE'] = 'window'
            t.meta['SACCCLSS'] = cls.window_type
            t.meta['SACCNAME'] = id(w)
            t.meta['EXTNAME'] = 'window:'+cls.window_type
            tables.append(t)
        return tables

    @classmethod
    def from_table(cls, table):
        return {table.meta['SACCNAME']: cls(table['values'], table['weight'])}
