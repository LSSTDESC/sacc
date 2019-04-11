import numpy as np
from astropy.table import Table
from .utils import Namespace

class BaseTracer:
    """
    A class representing some kind of tracer of astronomical objects.

    Generically, SACC2 data points correspond to some combination of tracers
    for example, tomographic two-point data has two tracers for each data
    point, indicating the n(z) for the corresponding tomographic bin.

    All Tracer objects have at least a name attribute.  Different
    subclassses have other requirements.  For example, n(z) tracers
    require z and n(z) arrays.

    In general you don't need to create tracer objects yourself -
    the Sacc2.add_tracer method will construct them for you.
    """
    _tracer_classes = {}

    def __init__(self, name):
        self.name = name

    def __init_subclass__(cls, tracer_type):
        cls._tracer_classes[tracer_type] = cls
        cls.tracer_type = tracer_type

    @classmethod
    def make(cls, tracer_type, name, *args, **kwargs):
        """
        Select a Tracer subclass based on tracer_type
        and instantiate in instance of it with the remaining
        arguments.

        Parameters
        ----------
        tracer_type: str
            Must correspond to the tracer_type of a subclass
        name: str
            The name for this specific tracer, e.g. a
            tomographic bin identifier.

        Returns
        -------
        instance: Tracer object
            An instance of a Tracer subclass
        """
        subclass = cls._tracer_classes[tracer_type]
        return subclass(name, *args, **kwargs)

    @classmethod
    def to_tables(cls, instance_list):
        """Convert a list of tracers to a list of astropy tables

        This is used when saving data to a file.

        This class method converts a list of tracers, each of which
        can instances of any subclass of BaseTracer, and turns them
        into a list of astropy tables, ready to be saved to disk.

        Some tracers generate a single table for all of the
        different instances, and others generate one table per
        instance.
    
        Parameters
        ----------
        instance_list: list
            List of tracer instances

        Returns
        -------

        tables: list
            List of astropy tables
        """
        tables = []
        for name, subcls in cls._tracer_classes.items():
            tracers = [t for t in instance_list if type(t) == subcls]
            tables += subcls.to_tables(tracers)
        return tables

    @classmethod
    def from_tables(cls, table_list):
        """Convert a list of astropy tables into a dictionary of tracers

        This is used when loading data from a file.

        This class method takes a list of tracers, such as those
        read from a file, and converts them into a list of instances.

        It is not quite the inverse of the to_tables method, since it
        returns a dict instead of a list.

        Parameters
        ----------
        table_list: list
            List of astropy tables

        Returns
        -------
        tracers: dict
            Dict mapping string names to tracer objects.

        """
        tracers = {}
        for table in table_list:
            subclass_name = table.meta['SACCCLSS']
            subclass = cls._tracer_classes[subclass_name]
            tracers.update(subclass.from_table(table))
        return tracers


class MiscTracer(BaseTracer, tracer_type='misc'):
    """A Tracer type for miscellaneous other data points.

    MiscTracers do not have any attributes except for their
    name, so can be used for tagging external data, for example.

    Attributes
    ----------

    name: str
        The name of the tracer

    """

    def __init__(self, name):
        super().__init__(name)

    @classmethod
    def to_tables(cls, instance_list):
        """Convert a list of MiscTracer instances to a astropy tables.

        This is used when saving data to file.

        All the instances are converted to a single table, which is
        returned in a list with one element so that it can be used
        in combination with the parent.

        You can use the parent class to_tables class method to convert
        a mixed list of different tracer types.

        You shouldn't generally need to call this method directly.

        Parameters
        ----------

        instance_list: list
            list of MiscTracer objects

        Returns
        -------

        tables: list
            List containing one astropy table
        """
        cols = [[obj.name for obj in instance_list]]
        table = Table(data=cols, names=['name'])
        table.meta['SACCTYPE'] = 'tracer'
        table.meta['SACCCLSS'] = cls.tracer_type
        table.meta['EXTNAME'] = f'tracer:{cls.tracer_type}'
        return [table]

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table into a dictionary of instances


        """
        return {name: cls(name) for name in table['name']}


class NZTracer(BaseTracer, tracer_type='NZ'):
    """
    A Tracer type for tomographic n(z) data.

    Takes two arguments arrays of z and n(z)

    Attributes
    ----------

    z: array
        Redshift sample values
    nz: array
        Number density n(z) at redshift sample points.
    """

    def __init__(self, name, z, nz):
        """
        Create a tracer corresponding to a distribution in redshift n(z),
        for example of galaxies.

        Parameters
        ----------
        name: str
            The name for this specific tracer, e.g. a
            tomographic bin identifier.
        z: array
            Redshift sample values
        nz: array
            Number density n(z) at redshift sample points.

        Returns
        -------
        instance: NZTracer object
            An instance of this class
        """
        super().__init__(name)
        self.z = np.array(z)
        self.nz = np.array(nz)

    @classmethod
    def to_tables(cls, instance_list):
        """Convert a list of NZTracers to a list of astropy tables

        This is used when saving data to a file.

        One table is generated per tracer.

        Parameters
        ----------
        instance_list: list
            List of tracer instances

        Returns
        -------

        tables: list
            List of astropy tables
        """
        tables = []
        for tracer in instance_list:
            names = ['z', 'nz']
            cols = [tracer.z, tracer.nz]
            table = Table(data=cols, names=names)
            table.meta['SACCTYPE'] = 'tracer'
            table.meta['SACCCLSS'] = cls.tracer_type
            table.meta['SACCNAME'] = tracer.name
            table.meta['EXTNAME'] = f'tracer:{cls.tracer_type}:{tracer.name}'
            tables.append(table)
        return tables

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.

        A single tracer object is read from the table.

        Parameters
        ----------
        table: astropy.table.Table
            Must contain the appropriate data, for example as saved
            by to_table.

        Returns
        -------
        tracers: dict
            Dict mapping string names to tracer objects.
            Only contains one key/value pair for the one tracer.

        """
        name = table.meta['SACCNAME']
        z = table['z']
        nz = table['nz']
        return {name: cls(name, z, nz)}
