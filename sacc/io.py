from .utils import numpy_to_vanilla
from io import BytesIO
import inspect
from astropy.table import Table

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

            # Check that the class variable _sub_classes, which bases classes
            # use to register their subclasses, exists.
            if not hasattr(cls, '_sub_classes'):
                raise RuntimeError("Base subclasses of BaseIO must have a dictionary class variable _sub_classes, but"
                                  f" {cls.__name__} does not have one defined.")

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


        # We could probably be using an Abstract Base Class rather than doing this.
        # Then you wouldn't get a warning until instantiation. That might be good
        # or bad.
        if cls.storage_type == ONE_OBJECT_PER_TABLE:
            check_has_standard_method(cls, 'to_table')
            check_has_class_method(cls, 'from_table')

        elif cls.storage_type == ONE_OBJECT_MULTIPLE_TABLES:
            check_has_standard_method(cls, 'to_tables')
            check_has_class_method(cls, 'from_tables')

        elif cls.storage_type == MULTIPLE_OBJECTS_PER_TABLE:
            check_has_class_method(cls, 'to_table')
            check_has_class_method(cls, 'from_table')
        else:
            raise RuntimeError(f"Subclasses of BaseIO must have a class variable storage_type set to one of "
                               f"{ONE_OBJECT_PER_TABLE}, {MULTIPLE_OBJECTS_PER_TABLE}, or {ONE_OBJECT_MULTIPLE_TABLES}, "
                               f"but {cls.__name__} has {cls.storage_type}.")


        if type_name.lower() in cls._sub_classes:
            raise RuntimeError(f"Subclasses of BaseIO must have unique type_name, "
                               f"but {type_name} is already registered.")

        cls._sub_classes[type_name.lower()] = cls
        cls.type_name = type_name


def to_tables(category_dict):
    """Convert a dict of objects to a list of astropy tables

    This is used when saving data to a file.

    This class method converts a dict of objects, each of which
    can instances of any subclass of BaseIO, and turns them
    into a list of astropy tables, ready to be saved to disk.

    Some object types generate a single table for all of the
    different instances, and others generate one table per
    instance, and some others generate multiple tables
    for a single instance.

    The storage type of each class decides which of these it is.

    Parameters
    ----------
    category_: dict[str, dict[str, BaseIO]]
        Tracer instances by category (e.g. "tracer", "window", "covariance"), then name
        (e.g. "source_1")

    Returns
    -------
    tables: list
        List of astropy tables
    """
    from .data_types import DataPoint

    # This is the list of tables that we will build up and return
    tables = []

    # The top level category_dict is a dict mapping
    # general categories of data, each represented by a different
    # subclass of BaseIO, to a dict of further subclasses of that subclass.
    for category, instance_dict in category_dict.items():
        multi_object_tables = {}

        # We handle the "data" category separately, since it is a special case
        if category == 'data' or category == 'metadata':
            continue

        for name, obj in instance_dict.items():
            # Get the class of the instance
            cls = type(obj)

            # Check if the class is a subclass of BaseIO
            if not issubclass(cls, BaseIO):
                raise RuntimeError(f"Instance {obj} of type {cls.__name__} does not subclass BaseIO.")

            if obj.storage_type == ONE_OBJECT_PER_TABLE:
                # If the storage type is ONE_OBJECT_PER_TABLE, we expect
                # that the table will return a single instance of the class.
                # print(f"Saving {name} of type {cls.type_name} in category {category} to a single table.")
                table = obj.to_table()
                table.meta['SACCTYPE'] = category
                table.meta['SACCCLSS'] = cls.type_name
                table.meta['SACCNAME'] = name
                tables.append(table)

            elif obj.storage_type == MULTIPLE_OBJECTS_PER_TABLE:
                # If the storage type is MULTIPLE_OBJECTS_PER_TABLE then
                # we need to collect together all the instances of this
                # class and convert at the end
                key = (cls, name)
                if key not in multi_object_tables:
                    multi_object_tables[key] = []
                multi_object_tables[key].append(obj)

            elif obj.storage_type == ONE_OBJECT_MULTIPLE_TABLES:
                # If the storage type is ONE_OBJECT_MULTIPLE_TABLES, we expect
                # that the table will return a dict of tables
                # each in its own table.
                # print(f"Saving {name} of type {cls.type_name} in category {category} to multiple tables.")
                tabs = obj.to_tables()
                for part_name, table in tabs.items():
                    table.meta['SACCTYPE'] = category
                    table.meta['SACCCLSS'] = cls.type_name
                    table.meta["SACCNAME"] = name
                    table.meta['SACCPART'] = part_name
                    tables.append(table)
            else:
                raise RuntimeError(f"Storage type {cls.storage_type} for {cls.__name__} is not recognized.")

        # Now process the multi-object tables for this category
        for (cls, name), instance_list in multi_object_tables.items():
            # Convert the list of instances to a single table
            table = cls.to_table(instance_list)
            table.meta['SACCTYPE'] = category
            table.meta['SACCNAME'] = name
            table.meta['SACCCLSS'] = cls.type_name
            tables.append(table)

    # Handle data points separately, since they are a special case
    data = category_dict.get('data', [])
    lookups = category_dict.get('window', {})

    # Because lots of objects share the same window function
    # we map a code number for a window to the window object
    # when serializing.
    lookups = {'window': {v: k for k, v in lookups.items()}}
    data_tables = DataPoint.to_tables(data, lookups=lookups)

    for name, table in data_tables.items():
        table.meta['SACCTYPE'] = "data"
        table.meta['SACCNAME'] = name
        tables.append(table)

    # Also handle metadata separately. Could consider a metadata class
    # that subclasses BaseIO and dict?
    tables.append(metadata_to_table(category_dict.get('metadata', {})))


    return tables

def from_tables(table_list):
    """Convert a list of astropy tables into a dictionary of sacc objects.

    This is used when loading data from a file.

    This class method takes a list of astropy tables, typically read from
    a file, and converts them all into instances of BaseIO subclasses.

    Parameters
    ----------
    table_list: list
        List of astropy tables

    Returns
    -------
    objects: dict[Str, dict[str, BaseIO]]
        Dict mapping category names then object names to instances of BaseIO subclasses.
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
        if table_category == 'metadata':
            # This is a metadata table, which we treat as a special case.
            outputs[table_category] = table_to_metadata(table)
            continue

        table_class_name = table.meta['SACCCLSS'].lower()
        # The class that represents this specific subtype
        base_class = BaseIO._base_subclasses[table_category]
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


def astropy_buffered_fits_write(filename, hdu_list):
    # Write out data - do it to a buffer because astropy
    # metadata performance on some file systems is terrible.
    buf = BytesIO()
    hdu_list.writeto(buf)
    # Rewind and read the binary data we just wrote
    buf.seek(0)
    output_data = buf.read()
    # Write the binary data to the target file
    with open(filename, "wb") as f:
        f.write(output_data)


def is_class_method(method):
    return callable(method) and inspect.ismethod(method) and not inspect.isfunction(method)


def check_has_class_method(cls, method_name):
    """Check if a class has a class method with the given name."""
    method = getattr(cls, method_name, None)
    if method is None:
        raise RuntimeError(f"As a BaseIO subclass, {cls.__name__} should have a class method {method_name} defined.")

    if not is_class_method(method):
        raise RuntimeError(f"As a BaseIO subclass, {cls.__name__} has {method_name}, but it is not defined as a class method.")

def check_has_standard_method(cls, method_name):
    """Check if a class has a regular method with the given name."""
    method = getattr(cls, method_name, None)
    if method is None:
        raise RuntimeError(f"As a BaseIO subclass, {cls.__name__} should have a method {method_name} defined.")

    if not callable(method):
        raise RuntimeError(f"As a BaseIO subclass, class {cls.__name__} has {method_name}, but it is not a method.")

    if is_class_method(method):
        raise RuntimeError(f"As a BaseIO subclass, {cls.__name__} has {method_name}, but it is defined as a class method or something else like that")

def metadata_to_table(metadata):
    """
    Convert a metadata dict to an astropy table.

    Because astropy table columns must have a single type,
    we store each item in the metadata dict as a separate column.

    Parameters
    ----------
    metadata: dict
        Dictionary of metadata items, where each key is a string,
        and values are simple unstructured types (int, float, str, bool, etc.).

    Returns
    -------
    table: astropy.table.Table
        An astropy table with a single row, where each column corresponds
        to a key in the metadata dict, and the first (only) row values
        is the corresponding value.
    """
    # For typing reasons each key is a column in the table
    # and there is only one row.

    keys = list(metadata.keys())
    values = [numpy_to_vanilla(metadata[key]) for key in keys]
    table: Table = Table(rows=[values], names=keys)
    table.meta['SACCTYPE'] = "metadata"
    table.meta['SACCCLSS'] = "metadata"
    table.meta['SACCNAME'] = "metadata"
    return table

def table_to_metadata(table):
    """
    Convert an astropy table to a metadata dict.

    See metadata_to_table for the format expected.

    Parameters
    ----------
    table: astropy.table.Table
        An astropy table with a single row, where each column corresponds
        to a key in the metadata dict, and the first (only) row values
        is the corresponding value.

    Returns
    -------
    metadata: dict
        Dictionary of metadata items, where each key is a string,
        and values are simple unstructured types (int, float, str, bool, etc.).
    """
    metadata = {}
    for key in table.colnames:
        metadata[key] = numpy_to_vanilla(table[key][0])
    return metadata
