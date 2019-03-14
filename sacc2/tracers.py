
class Tracer:
    """
    A class representing some kind of tracer of astronomical objects.

    Generically, SACC2 data points correspond to some combination of tracers
    for example, tomographic two-point data has two tracers for each data
    point, indicating the n(z) for the corresponding tomographic bin.

    All Tracer objects have at least a name attribute.  Different
    subclasses have other requirements.  For example, n(z) tracers
    require z and n(z) arrays.

    In general you don't need to create tracer objects yourself - 
    the Sacc2.add_tracer method will construct them for you.
    """
    subclasses = {}
    def __init__(self, name):
        self.name = name

    def __init_subclass__(cls, tracer_type):
        cls.subclasses[tracer_type] = cls

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
        subclass = cls.subclasses[tracer_type]
        return subclass(name, *args, **kwargs)
        
class MiscTracer(Tracer, tracer_type='misc'):
    """
    A Tracer type for miscellaneous other data points
    """
    def __init__(self, name):
        super().__init__(name)

        
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
        self.z = z
        self.nz = nz
