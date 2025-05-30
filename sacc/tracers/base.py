from ..utils import Namespace, unique_list



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
        # We encourage people to use existing quantity names, and issue a
        # warning if they do not to prod them in the right direction.
        quantity = kwargs.pop('quantity', 'generic')
        self.name = name
        self.quantity = quantity
        self.metadata = kwargs.pop('metadata', {})

    def __init_subclass__(cls, tracer_type):
        cls._tracer_classes[tracer_type.lower()] = cls
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
            The name for this specific tracer.

        Returns
        -------
        instance: Tracer object
            An instance of a Tracer subclass
        """
        subclass = cls._tracer_classes[tracer_type.lower()]
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
            # If the list is empty, we don't want to append any tables.
            if tracers:
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

        Subclasses overrides of this method do the actual work, but
        should *NOT* call this parent base method.

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
        subclass_names = unique_list(table.meta['SACCCLSS'].lower()
                                     for table in table_list)
        subclasses = [cls._tracer_classes[name]
                      for name in subclass_names]

        # For each subclass find the tables representing that subclass.
        # We do it like this because we might want to represent one tracer with
        # multiple tables, or one table can have multiple tracers -
        # it depends on the tracers class and how complicated it is.
        for name, subcls in zip(subclass_names, subclasses):
            subcls_table_list = [table for table in table_list
                                 if table.meta['SACCCLSS'].lower() == name]
            # and ask the subclass to read from those tables.
            tracers.update(subcls.from_tables(subcls_table_list))
        return tracers

