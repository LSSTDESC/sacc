import numpy as np
try:
    from qp.core.ensemble import Ensemble
except:
    Ensemble = None
from ..utils import Namespace
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

    def __eq__(self, other):
        """"
        Compare two Tracers for equality.

        Tracers can only be equal if they have the same type.
        Tracers of the same type are equal if they have all the same
        attribute values.

        There is no need to override this method in subclasses.

        Parameters
        ----------
        other: Tracer
            The other Tracer to compare against.

        Returns
        -------
        value: bool
            True if the two Tracers are equal, else False
        """
        if not isinstance(other, self.__class__):
            return NotImplemented  # not False, to help ensure symmetry
        vars1, vars2 = vars(self), vars(other)

        if len(vars1) != len(vars2):
            return False

        varnames1 = set(vars1.keys())
        varnames2 = set(vars2.keys())
        if varnames1 != varnames2:
            return False

        for name in varnames1:
            v1 = vars1[name]
            v2 = vars2[name]
            # TODO: Remove this work-around one we have a way to test Ensembles for equality.
            # If we do not have qp installed, then we do not attempt to compare Ensembles.
            if Ensemble is not None and isinstance(v1, Ensemble):
                continue
            try:
                if v1 != v2:
                    return False
            except ValueError:  # raised by numpy arrays
                if not np.allclose(v1, v2):
                    return False
        return True


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
