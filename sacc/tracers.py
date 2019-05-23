import numpy as np
from astropy.table import Table
from .utils import Namespace, hide_null_values, remove_dict_null_values, unique_list

class BaseTracer:
    """
    A class representing some kind of tracer of astronomical objects.

    Generically, SACC data points correspond to some combination of tracers
    for example, tomographic two-point data has two tracers for each data
    point, indicating the n(z) for the corresponding tomographic bin.

    All Tracer objects have at least a name attribute.  Different
    subclassses have other requirements.  For example, n(z) tracers
    require z and n(z) arrays.

    In general you don't need to create tracer objects yourself -
    the Sacc.add_tracer method will construct them for you.
    """
    _tracer_classes = {}

    def __init__(self, name, **kwargs):
        self.name = name
        self.metadata = kwargs.pop('metadata', {})

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
        obj = subclass(name, *args, **kwargs)
        return obj

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

        Subclasses overrides of this method do the actual work, but should *NOT* call
        this parent base method.

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
        # Figure out the different subclasses that are present
        subclass_names = unique_list(table.meta['SACCCLSS'] for table in table_list)
        subclasses = [cls._tracer_classes[name] for name in subclass_names]

        # For each subclass find the tables representing that subclass.
        # We do it like this because we might want to represent one tracer with
        # multiple tables, or one table can have multiple tracers - it depends on
        # the tracers class and how complicated it is.
        for name, subcls in zip(subclass_names, subclasses):
            subcls_table_list = [table for table in table_list if table.meta['SACCCLSS']==name]
            # and ask the subclass to read from those tables.
            tracers.update(subcls.from_tables(subcls_table_list))
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

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

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
        metadata_cols = set()
        for obj in instance_list:
            metadata_cols.update(obj.metadata.keys())
        metadata_cols = list(metadata_cols)

        cols = [[obj.name for obj in instance_list]]
        for name in metadata_cols:
            cols.append([obj.metadata.get(name) for obj in instance_list])

        table = Table(data=cols, names=['name']+ metadata_cols)
        table.meta['SACCTYPE'] = 'tracer'
        table.meta['SACCCLSS'] = cls.tracer_type
        table.meta['EXTNAME'] = f'tracer:{cls.tracer_type}'
        hide_null_values(table)
        return [table]

    @classmethod
    def from_tables(cls, table_list):
        """Convert a list of astropy table into a dictionary of MiscTracer instances.

        In general table_list should have a single element in, since all the 
        MiscTracers are stored in a single table during to_tables

        Parameters
        ----------
        table_list: List[astropy.table.Table]

        Returns
        -------
        tracers: Dict[str: MiscTracer]
        """
        tracers = {}

        for table in table_list:
            metadata_cols = [col for col in table.colnames if col != 'name']
            
            for row in table:
                name = row['name']
                metadata = {key: row[key] for key in metadata_cols}
                remove_dict_null_values(metadata)
                tracers[name] = cls(name, metadata=metadata)
        return tracers


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
    extra_columns: dict[str: array] or dict[int:array]
        Additional estimates of the same n(z), by name
    """

    def __init__(self, name, z, nz, extra_columns=None, **kwargs):
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
        extra_columns: dict[str:array]
            Optional, default=None.  Additional realizations or
            estimates of the same n(z), by name.

        Returns
        -------
        instance: NZTracer object
            An instance of this class
        """
        super().__init__(name, **kwargs)
        self.z = np.array(z)
        self.nz = np.array(nz)
        self.extra_columns = {} if extra_columns is None else extra_columns

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
            for nz_id, col in tracer.extra_columns.items():
                names.append(str(nz_id))
                cols.append(col)
            table = Table(data=cols, names=names)
            table.meta['SACCTYPE'] = 'tracer'
            table.meta['SACCCLSS'] = cls.tracer_type
            table.meta['SACCNAME'] = tracer.name
            table.meta['EXTNAME'] = f'tracer:{cls.tracer_type}:{tracer.name}'
            for key, value in tracer.metadata.items():
                table.meta['META_'+key] = value
            remove_dict_null_values(table.meta)
            tables.append(table)
        return tables

    @classmethod
    def from_tables(cls, table_list):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        A single tracer object is read from the table.

        Parameters
        ----------
        table_list: list[astropy.table.Table]
            Must contain the appropriate data, for example as saved
            by to_table.

        Returns
        -------
        tracers: dict
            Dict mapping string names to tracer objects.
            Only contains one key/value pair for the one tracer.

        """
        tracers = {}
        for table in table_list:
            name = table.meta['SACCNAME']
            z = table['z']
            nz = table['nz']
            extra_columns = {}
            for col in table.columns.values():
                if col.name not in ['z', 'nz']:
                    extra_columns[col.name] = col.data

            metadata = {}
            for key, value in table.meta.items():
                if key.startswith("META_"):
                    metadata[key[5:]] = value
            tracers[name] = cls(name, z, nz, extra_columns=extra_columns, metadata=metadata)
        return tracers
