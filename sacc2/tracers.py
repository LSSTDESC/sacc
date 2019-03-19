import numpy as np
from astropy.table import Table
from astropy.io import fits

class Tracer:
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

    def to_dict(self):
        return {
            "type": self.tracer_type,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d):
        # subclasses must not call the parent implementation of this method!
        subclass_name = d['type']
        subclass = cls._tracer_classes[subclass_name]
        return subclass.from_dict(d)

    def _add_fits_info(self, hdu):
        hdu.name = self.name
        hdu.header['sacctype'] = 'tracer'
        hdu.header['saccname'] = self.name
        hdu.header['saccclss'] = self.tracer_type
        return hdu

    @classmethod
    def from_fits(cls, hdu):
        subclass_name = hdu.header['saccclss']
        subclass = cls._tracer_classes[subclass_name]
        return subclass.from_fits(hdu)
        
class MiscTracer(Tracer, tracer_type='misc'):
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
        name = hdu.header['saccname']
        return cls(name)

    def to_fits(self):
        hdu = fits.BinTableHDU()
        self._add_fits_info(hdu)
        return hdu



        
class NZTracer(Tracer, tracer_type='NZ'):
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

    def to_dict(self):
        d = super().to_dict()
        d['z'] = self.z.tolist()
        d['nz'] = self.nz.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d['name'], d['z'], d['nz'])

    def to_fits(self):
        tab = Table(data={'z':self.z, 'nz':self.nz})
        hdu = fits.table_to_hdu(tab)
        self._add_fits_info(hdu)
        return hdu

    @classmethod
    def from_fits(cls, hdu):
        name = hdu.header['saccname']
        z = hdu.data['z']
        nz = hdu.data['nz']
        return cls(name, z, nz)

