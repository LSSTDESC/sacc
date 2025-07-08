from ..utils import Namespace, unique_list
from ..io import BaseIO, ONE_OBJECT_PER_TABLE, MULTIPLE_OBJECTS_PER_TABLE, ONE_OBJECT_MULTIPLE_TABLES


standard_quantities = Namespace('galaxy_shear',
                                'galaxy_density',
                                'galaxy_convergence',
                                'cluster_density',
                                'cmb_temperature',
                                'cmb_polarization',
                                'cmb_convergence',
                                'cmb_tSZ',
                                'cmb_kSZ',
                                'cluster_mass_count_wl',
                                'cluster_mass_count_xray',
                                'cluster_mass_count_tSZ',
                                'generic')


class BaseTracer(BaseIO):
    """
    A class representing some kind of tracer of astronomical objects.

    Generically, SACC data points correspond to some combination of tracers
    for example, tomographic two-point data has two tracers for each data
    point, indicating the n(z) for the corresponding tomographic bin.

    Concrete subclasses must implement to_tables and from_tables methods

    All Tracer objects have at least a name attribute.  Different
    subclassses have other requirements.  For example, n(z) tracers
    require z and n(z) arrays.

    In general you don't need to create tracer objects yourself -
    the Sacc.add_tracer method will construct them for you.
    """
    _sub_classes = {}

    @property
    def tracer_type(self):
        """Legacy value - this name has been replaced by type_name."""
        return self.type_name

    def __init__(self, name, **kwargs):
        # We encourage people to use existing quantity names, and issue a
        # warning if they do not to prod them in the right direction.
        quantity = kwargs.pop('quantity', 'generic')
        self.name = name
        self.quantity = quantity
        self.metadata = kwargs.pop('metadata', {})


    @classmethod
    def make(cls, tracer_type, name, *args, **kwargs):
        """
        Select a Tracer subclass based on tracer_type
        and instantiate in instance of it with the remaining
        arguments.

        Parameters
        ----------
        tracer_type: str
            Must correspond to the type_name of a subclass

        name: str
            The name for this specific tracer.

        Returns
        -------
        instance: Tracer object
            An instance of a Tracer subclass
        """
        subclass = cls._sub_classes[tracer_type.lower()]
        obj = subclass(name, *args, **kwargs)
        return obj
