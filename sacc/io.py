from .utils import unique_list
import numpy as np


ONE_OBJECT_PER_TABLE = "ONE_OBJECT_PER_TABLE"
MULTIPLE_OBJECTS_PER_TABLE = "MULTIPLE_OBJECTS_PER_TABLE"
ONE_OBJECT_MULTIPLE_TABLES = "ONE_OBJECT_MULTIPLE_TABLES"


"""
if storage_type == ONE_OBJECT_PER_TABLE then class must have
- to_table returning a single astropy table from an instance
- from_table returning a single instance from an astropy table

* tables must have a unique name

if storage_type == MULTIPLE_OBJECTS_PER_TABLE then the class must have
- to_table return one table from a list of instances
- from_table returning a list of instances from a single astropy table


if storage_type == ONE_OBJECT_MULTIPLE_TABLES then the class must have
- to_tables returning a list of astropy tables from a single instance
- from_tables returning a single instance from a list of astropy tables

* name base can be shared between tables

"""


class BaseIO:
    """
    This base class represents interfaces for input/output operations
    in which:
    - subclasses define methods to_tables and from_tables
    - to_tables converts a list of instances into a list of astropy tables
    - from_tables converts a list of astropy tables into a dictionary of instances

    The reason for this design is efficiency in packing together tracer data into
    a smaller number of tables. For some tracer types we want to store many tracers
    in a single table, while for others we want to store each tracer in its own table.

    New hierarchies of objects can be created 
    """
    _base_subclasses = {}
    storage_type = "NOT_DEFINED"

    def __init_subclass__(cls, type_name=""):

        # We can have base subclasses that do not have a type name
        if cls.__name__.startswith('Base'):
            BaseIO._base_subclasses[cls.__name__[4:].lower()] = cls
            return
        
        if type_name == "":
            raise RuntimeError("Subclasses that use the table IO system like tracers must have a type_name set when defining them.")
        
        # Check that the storage_type is defined and valid
        if cls.storage_type == "NOT_DEFINED":
            raise RuntimeError(f"Subclasses of BaseIO must define a class variable storage_type, but {cls.__name__} does not have one defined.")
        
        if cls.storage_type not in (ONE_OBJECT_PER_TABLE, MULTIPLE_OBJECTS_PER_TABLE, ONE_OBJECT_MULTIPLE_TABLES):
            raise RuntimeError(f"Subclasses of BaseIO must have a class variable storage_type set to one of "
                               f"{ONE_OBJECT_PER_TABLE}, {MULTIPLE_OBJECTS_PER_TABLE}, or {ONE_OBJECT_MULTIPLE_TABLES}, "
                               f"but {cls.__name__} has {cls.storage_type}.")


        # Check that the class variable _sub_classes, which bases classes
        # use to register their subclasses, exists.
        if not hasattr(cls, '_sub_classes'):
            raise RuntimeError("Subclasses of BaseIO must have a dictionary class variable _sub_classes")
        
        if type_name.lower() in cls._sub_classes:
            raise RuntimeError(f"Subclasses of BaseIO must have unique type_name, "
                               f"but {type_name} is already registered.")

        cls._sub_classes[type_name.lower()] = cls
        cls.type_name = type_name


    @classmethod
    def to_tables(cls, category_dict):
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
        category_: dict[str, dict[str, BaseIO]]
            Tracer instances by category, then name

        Returns
        -------
        tables: list
            List of astropy tables
        """
        tables = []
        multi_object_tables = {}
        data_tables = []
        for category, instance_dict in category_dict.items():
            if category == 'data':
                data_tables.append(instance_dict)
                continue
            for name, obj in instance_dict.items():
                # Get the class of the instance
                cls = type(name, obj)

                # Check if the class is a subclass of BaseIO
                if not issubclass(cls, BaseIO):
                    raise RuntimeError(f"Instance {name, obj} of type {cls.__name__} does not subclass BaseIO.")

                # Convert the instance to tables using its own method
                if obj.storage_type == ONE_OBJECT_PER_TABLE:
                    # If the storage type is ONE_OBJECT_PER_TABLE, we expect
                    # that the table will return a single instance of the class.
                    table = obj.to_table()
                    table.meta['SACCTYPE'] = category
                    table.meta['SACCCLSS'] = cls.type_name
                    table.meta['SACCNAME'] = name
                    tables.append(table)
                elif obj.storage_type == MULTIPLE_OBJECTS_PER_TABLE:
                    # If the storage type is MULTIPLE_OBJECTS_PER_TABLE then
                    # we need to collect together all the instances of this
                    # class and convert at the end
                    if cls not in multi_object_tables:
                        multi_object_tables[cls] = []
                    multi_object_tables[cls].append(obj)
                elif obj.storage_type == ONE_OBJECT_MULTIPLE_TABLES:
                    # If the storage type is ONE_OBJECT_MULTIPLE_TABLES, we expect
                    # that the table will return a dict of instances of the class,
                    # each in its own table.
                    tables = obj.to_tables()
                    for name, table in tables.items():
                        table.meta['SACCTYPE'] = category
                        table.meta['SACCCLSS'] = cls.type_name
                        table.meta['SACCNAME'] = obj.name
                        table.meta['SACCPART'] = name
                else:
                    raise RuntimeError(f"Storage type {cls.storage_type} for {cls.__name__} is not recognized.")

        # Now process the multi-object tables
        for cls, instance_list in multi_object_tables.items():
            # Convert the list of instances to a single table
            table = cls.to_table(instance_list)
            tables.append(table)

        for dt in data_tables:
            # each table has a bunch of data points in.

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
        from .data_types import DataPoint
        outputs = {}
        multi_tables = {}
        data_point_tables = []

        for table in table_list:
            # what general category of object is this table, e.g.
            # tracers, windows, data points.
            table_category = table.meta['SACCTYPE'].lower()

            # what specific subclass of that category is this table?
            # e.g. N(z) tracer, top hat window, etc.
            if table_category == 'data':
                # This is a data table, which we treat as a special case.
                # because the ordering here is particularly important
                table_class_name = "datapoint"
                table_class = DataPoint
                data_point_tables.append(table)
                continue

            else:
                table_class_name = table.meta['SACCCLSS'].lower()
                # The class that represents this specific subtype
                base_class = cls._base_subclasses[table_category]
                table_class = base_class._sub_classes[table_class_name]
                if table_category not in outputs:
                    outputs[table_category] = {}

            # We will be doing the types where an object is split up
            # over multiple tables separately, so we store them
            # in a dict for later processing.
            if table_class.storage_type == ONE_OBJECT_MULTIPLE_TABLES:
                name = table.meta["SACCNAME"]
                if "SACCPART" in table.meta:
                    part = table.meta["SACCPART"]
                else:
                    # legacy tables may not have the part - in this case
                    # name should hopefully be kind:class:part
                    part = table.meta['EXTNAME'].rsplit(":", 1)[-1]
                key = (table_category, table_class, name)
                if key not in multi_tables:
                    multi_tables[key] = {}
                multi_tables[key][part] = table
                continue

            # # Delete this later once things are working:
            # print(f"Processing table {table.meta['EXTNAME']} of type {table_class_name} in category {table_category} storage type {table_class.storage_type} length {len(table)}")

            # Convert the tables into either one instance of the class or a list of instances
            if table_class.storage_type == ONE_OBJECT_PER_TABLE:
                # If the storage type is ONE_OBJECT_PER_TABLE, we expect
                # that the table will return a single instance of the class.
                obj = table_class.from_table(table)
                name = table.meta['SACCNAME']
                outputs[table_category][name] = obj

            elif table_class.storage_type == MULTIPLE_OBJECTS_PER_TABLE:
                # If the storage type is MULTIPLE_OBJECTS_PER_TABLE, we expect
                # that the table will return a dict of instances of the class,
                # keyed by their names.
                objs = table_class.from_table(table)
                outputs[table_category].update(objs)

        # Now process the multi-table objects that we collected above.
        for key, m_tables in multi_tables.items():
            # key is a tuple of (table_category, table_class, name)
            table_category, table_class, name = key
            # Convert the dict of tables into a single instance
            obj = table_class.from_tables(m_tables)
            outputs[table_category][name] = obj

        # Now finally process the data point tables.
        data_points = []
        if 'window' in outputs:
            lookups = {'window': outputs['window']}
        else:
            lookups = {}
        for table in data_point_tables:
            # Each data point table is a single data point
            dps = DataPoint.from_table(table, lookups=lookups)
            data_points.extend(dps)

        outputs['data'] = data_points
        return outputs


def numpy_to_vanilla(x):
    if type(x) == np.str_:
        x = str(x)
    elif type(x) == np.int64:
        x = int(x)
    elif type(x) == np.float64:
        x = float(x)
    return x