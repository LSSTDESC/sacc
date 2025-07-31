from .base import BaseTracer, MULTIPLE_OBJECTS_PER_TABLE
from astropy.table import Table
from ..utils import hide_null_values, remove_dict_null_values

class MiscTracer(BaseTracer, type_name='Misc'):
    """A Tracer type for miscellaneous other data points.

    MiscTracers do not have any attributes except for their
    name, so can be used for tagging external data, for example.

    Parameters
    ----------
    name: str
        The name of the tracer
    """
    storage_type = MULTIPLE_OBJECTS_PER_TABLE

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    @classmethod
    def to_table(cls, instance_list):
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

        cols = [[obj.name for obj in instance_list],
                [obj.quantity for obj in instance_list]]
        for name in metadata_cols:
            cols.append([obj.metadata.get(name) for obj in instance_list])

        table = Table(data=cols,
                      names=['name', 'quantity'] + metadata_cols)
        hide_null_values(table)
        return table

    @classmethod
    def from_table(cls, table):
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

        metadata_cols = [col for col in table.colnames
                            if col not in ['name', 'quantity']]

        for row in table:
            name = row['name']
            quantity = row['quantity']
            metadata = {key: row[key] for key in metadata_cols}
            remove_dict_null_values(metadata)
            tracers[name] = cls(name, quantity=quantity, metadata=metadata)
        return tracers
