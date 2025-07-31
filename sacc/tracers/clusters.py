from .base import BaseTracer, MULTIPLE_OBJECTS_PER_TABLE
from astropy.table import Table

class BinZTracer(BaseTracer, type_name="bin_z"):  # type: ignore
    """A tracer for a single redshift bin. The tracer shall
    be used for binned data where we want a desired quantity
    per interval of redshift, such that we only need the data
    for a given interval instead of at individual redshifts."""

    storage_type = MULTIPLE_OBJECTS_PER_TABLE

    def __init__(self, name: str, lower: float, upper: float, **kwargs):
        """
        Create a tracer corresponding to a single redshift bin.

        :param name: The name of the tracer
        :param lower: The lower bound of the redshift bin
        :param upper: The upper bound of the redshift bin
        """
        super().__init__(name, **kwargs)
        self.lower = lower
        self.upper = upper

    def __eq__(self, other) -> bool:
        """Test for equality.  If :python:`other` is not a
        :python:`BinZTracer`, then it is not equal to :python:`self`.
        Otherwise, they are equal if names, and the z-range of the bins,
        are equal."""
        if not isinstance(other, BinZTracer):
            return False
        return (
            self.name == other.name
            and self.lower == other.lower
            and self.upper == other.upper
        )

    @classmethod
    def to_table(cls, instance_list):
        """Convert a list of BinZTracers to a single astropy table

        This is used when saving data to a file.
        One table is generated with the information for all the tracers.

        :param instance_list: List of tracer instances
        :return: List with a single astropy table
        """

        names = ["name", "quantity", "lower", "upper"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.lower for obj in instance_list],
            [obj.upper for obj in instance_list],
        ]

        table = Table(data=cols, names=names)
        table.meta["SACCTYPE"] = "tracer"
        table.meta["SACCCLSS"] = cls.type_name
        table.meta["EXTNAME"] = f"tracer:{cls.type_name}"
        return table

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        One tracer object is created for each "row" in each table.

        :param table_list: List of astropy tables
        :return: Dictionary of tracers
        """
        tracers = {}

        for row in table:
            name = row["name"]
            quantity = row["quantity"]
            lower = row["lower"]
            upper = row["upper"]
            tracers[name] = cls(name, quantity=quantity, lower=lower, upper=upper)
        return tracers

class BinLogMTracer(BaseTracer, type_name="bin_logM"):  # type: ignore
    """A tracer for a single log-mass bin. The tracer shall
    be used for binned data where we want a desired quantity
    per interval of log(mass), such that we only need the data
    for a given interval instead of at individual masses."""
    storage_type = MULTIPLE_OBJECTS_PER_TABLE

    def __init__(self, name: str, lower: float, upper: float, **kwargs):
        """
        Create a tracer corresponding to a single log-mass bin.

        :param name: The name of the tracer
        :param lower: The lower bound of the log-mass bin
        :param upper: The upper bound of the log-mass bin
        """
        super().__init__(name, **kwargs)
        self.lower = lower
        self.upper = upper

    def __eq__(self, other) -> bool:
        """Test for equality.  If :python:`other` is not a
        :python:`BinLogMTracer`, then it is not equal to :python:`self`.
        Otherwise, they are equal if names, and the z-range of the bins,
        are equal."""
        if not isinstance(other, BinLogMTracer):
            return False
        return (
            self.name == other.name
            and self.lower == other.lower
            and self.upper == other.upper
        )

    @classmethod
    def to_table(cls, instance_list):
        """Convert a list of BinLogMTracers to a single astropy table

        This is used when saving data to a file.
        One table is generated with the information for all the tracers.

        :param instance_list: List of tracer instances
        :return: List with a single astropy table
        """

        names = ["name", "quantity", "lower", "upper"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.lower for obj in instance_list],
            [obj.upper for obj in instance_list],
        ]
        table = Table(data=cols, names=names)
        table.meta["SACCTYPE"] = "tracer"
        table.meta["SACCCLSS"] = cls.type_name
        table.meta["EXTNAME"] = f"tracer:{cls.type_name}"
        return table

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        One tracer object is created for each "row" in each table.

        :param table_list: List of astropy tables
        :return: Dictionary of tracers
        """
        tracers = {}

        for row in table:
            name = row["name"]
            quantity = row["quantity"]
            lower = row["lower"]
            upper = row["upper"]
            tracers[name] = cls(name, quantity=quantity, lower=lower, upper=upper)
        return tracers


class BinRichnessTracer(BaseTracer, type_name="bin_richness"):  # type: ignore
    """A tracer for a single richness bin. The tracer shall
    be used for binned data where we want a desired quantity
    per interval of log(richness), such that we only need the data
    for a given interval instead of at individual richness."""
    storage_type = MULTIPLE_OBJECTS_PER_TABLE

    def __eq__(self, other) -> bool:
        """Test for equality. If :python:`other` is not a
        :python:`BinRichnessTracer`, then it is not equal to :python:`self`.
        Otherwise, they are equal if names and the richness-range of the
        bins, are equal."""
        if not isinstance(other, BinRichnessTracer):
            return False
        return (
            self.name == other.name
            and self.lower == other.lower
            and self.upper == other.upper
        )

    def __init__(self, name: str, lower: float, upper: float, **kwargs):
        """
        Create a tracer corresponding to a single richness bin.

        :param name: The name of the tracer
        :param lower: The lower bound of the richness bin in log10.
        :param upper: The upper bound of the richness bin in log10.
        """
        super().__init__(name, **kwargs)
        self.lower = lower
        self.upper = upper

    @classmethod
    def to_table(cls, instance_list):
        """Convert a list of BinZTracers to a list of astropy tables

        This is used when saving data to a file.
        One table is generated with the information for all the tracers.

        :param instance_list: List of tracer instances
        :return: List with a single astropy table
        """
        names = ["name", "quantity", "lower", "upper"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.lower for obj in instance_list],
            [obj.upper for obj in instance_list],
        ]

        table = Table(data=cols, names=names)
        table.meta["SACCTYPE"] = "tracer"
        table.meta["SACCCLSS"] = cls.type_name
        table.meta["EXTNAME"] = f"tracer:{cls.type_name}"
        return table

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        One tracer object is created for each "row" in each table.

        :param table_list: List of astropy tables
        :return: Dictionary of tracers
        """
        tracers = {}

        for row in table:
            name = row["name"]
            quantity = row["quantity"]
            lower = row["lower"]
            upper = row["upper"]
            tracers[name] = cls(
                name,
                quantity=quantity,
                lower=lower,
                upper=upper,
            )
        return tracers


class BinRadiusTracer(BaseTracer, type_name="bin_radius"):  # type: ignore
    """A tracer for a single radial bin, e.g. when dealing with cluster shear profiles.
    It gives the bin edges and the value of the bin "center". The latter would typically
    be returned by CLMM and correspond to the average radius of the galaxies in that
    radial bin. """

    storage_type = MULTIPLE_OBJECTS_PER_TABLE

    def __eq__(self, other) -> bool:
        """Test for equality. If :python:`other` is not a
        :python:`BinRadiusTracer`, then it is not equal to :python:`self`.
        Otherwise, they are equal if names and the r-range and centers of the
        bins, are equal."""
        if not isinstance(other, BinRadiusTracer):
            return False
        return (
            self.name == other.name
            and self.lower == other.lower
            and self.center == other.center
            and self.upper == other.upper
        )

    def __init__(self, name: str, lower: float, upper: float, center: float, **kwargs):
        """
        Create a tracer corresponding to a single radial bin.

        :param name: The name of the tracer
        :param lower: The lower bound of the radius bin
        :param upper: The upper bound of the radius bin
        :param center: The value to use if a single point-estimate is needed.

        Note that :python:`center` need not be the midpoint between
        :python:`lower` and :python:`upper`'.
        """
        super().__init__(name, **kwargs)
        self.lower = lower
        self.upper = upper
        self.center = center

    @classmethod
    def to_table(cls, instance_list):
        """Convert a list of BinRadiusTracers to a single astropy table

        This is used when saving data to a file.
        One table is generated with the information for all the tracers.

        :param instance_list: List of tracer instances
        :return: List with a single astropy table
        """

        names = ["name", "quantity", "lower", "upper", "center"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.lower for obj in instance_list],
            [obj.upper for obj in instance_list],
            [obj.center for obj in instance_list],
        ]

        table = Table(data=cols, names=names)
        return table

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        One tracer object is created for each "row" in each table.

        :param table_list: List of astropy tables
        :return: Dictionary of tracers
        """
        tracers = {}

        for row in table:
            name = row["name"]
            quantity = row["quantity"]
            lower = row["lower"]
            upper = row["upper"]
            center = row["center"]
            tracers[name] = cls(
                name,
                quantity=quantity,
                lower=lower,
                upper=upper,
                center=center,
            )
        return tracers
