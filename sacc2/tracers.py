import numpy as np
from astropy.table import Table
from astropy.io import fits

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
        tables = []
        for name, subcls in cls._tracer_classes.items():
            tracers = [t for t in instance_list if type(t)==subcls]
            tables += subcls.to_tables(tracers)
        return tables


    @classmethod
    def from_tables(cls, table_list):
        tracers = {}
        for table in table_list:
            subclass_name = table.meta['SACCCLSS']
            subclass = cls._tracer_classes[subclass_name]
            tracers.update(subclass.from_table(table))
        return tracers




class MiscTracer(BaseTracer, tracer_type='misc'):
    """
    A Tracer type for miscellaneous other data points
    """
    def __init__(self, name):
        super().__init__(name)

    @classmethod
    def from_dict(cls, d):
        return cls(d['name'])

    @classmethod
    def from_fits(cls, hdu):
        name = hdu.header['SACCNAME']
        return cls(name)

    @classmethod
    def to_tables(cls, instance_list):
        cols = [[obj.name for obj in instance_list]]
        table = Table(data=cols, names=['name'])
        table.meta['SACCTYPE'] = 'tracer'
        table.meta['SACCCLSS'] = cls.tracer_type
        table.meta['EXTNAME'] = f'tracer:{cls.tracer_type}'
        return [table]

    @classmethod
    def from_table(cls, table):
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
        name = table.meta['SACCNAME']
        z = table['z']
        nz = table['nz']
        return {name: cls(name, z, nz)}
