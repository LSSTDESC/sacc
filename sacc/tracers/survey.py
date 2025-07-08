from .base import BaseTracer, MULTIPLE_OBJECTS_PER_TABLE
from astropy.table import Table


class SurveyTracer(BaseTracer, type_name="survey"):  # type: ignore
    """A tracer for the survey definition. It shall 
    be used to filter data related to a given survey
    and to provide the survey sky-area of analysis."""

    storage_type = MULTIPLE_OBJECTS_PER_TABLE

    def __eq__(self, other) -> bool:
        """Test for equality. If :python:`other` is not a
        :python:`SurveyTracer`, then it is not equal to :python:`self`.
        Otherwise, they are equal if names and the sky-areas are equal."""
        if not isinstance(other, SurveyTracer):
            return False
        return self.name == other.name and self.sky_area == other.sky_area

    def __init__(self, name: str, sky_area: float, **kwargs):
        """
        Create a tracer corresponding to the survey definition.

        :param name: The name of the tracer
        :param sky_area: The survey's sky area in square degrees
        """
        super().__init__(name, **kwargs)
        self.sky_area = sky_area

    @classmethod
    def to_table(cls, instance_list):
        """Convert a list of SurveyTracer to a list of astropy tables

        This is used when saving data to a file.
        One table is generated with the information for all the tracers.

        :param instance_list: List of tracer instances
        :return: List of astropy tables with one table
        """
        names = ["name", "quantity", "sky_area"]

        cols = [
            [obj.name for obj in instance_list],
            [obj.quantity for obj in instance_list],
            [obj.sky_area for obj in instance_list],
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
            sky_area = row["sky_area"]
            tracers[name] = cls(
                name,
                quantity=quantity,
                sky_area=sky_area,
            )
        return tracers
