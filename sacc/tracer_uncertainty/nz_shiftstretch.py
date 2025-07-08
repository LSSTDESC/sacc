from .base import BaseTracerUncertainty
from ..io import ONE_OBJECT_PER_TABLE
from astropy.table import Table
import numpy as np


class NZShiftStretchUncertainty(BaseTracerUncertainty, type_name="nz_ShiftStretch"):
    """
    Class to handle uncertainty in number density tracers.
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, name, tracer_names, ShiftStretch_mean, ShiftStretch_cov):
        """
        Initialize the NZShiftStretchUncertainty object.
        Parameters
        ----------
        tracer_names : list of str
            List of tracer names to which the uncertainty applies.
        ShiftStretch_mean : array-like
            Mean ShiftStretch values for the tracers.
            This is strucutred as (shift_1, stretch_1, shift_2, stretch_2, ...).
        ShiftStretch_cov : array-like
            covariance of the ShiftStretch values for the tracers.
            Follows the ordering of the mean
        """

        super().__init__(name, tracer_names)
        self.ShiftStretch_mean = np.array(ShiftStretch_mean)
        self.ShiftStretch_cov = np.array(ShiftStretch_cov)
        self.nparams = 2

    @classmethod
    def from_table(self, table):
        """
        Read an NZShiftStretchUncertainty object from an astropy table.

        Parameters
        ----------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.

        Returns
        -------
        instance: NZShiftStretchUncertainty
            An instance of the NZShiftStretchUncertainty class.
        """
        mean = table["ShiftStretch_mean"]
        cov = table["ShiftStretch_cov"]
        tracer_names = [table.meta[f"TRACER_NAME_{i}"] for i in range(table.meta["N_TRACERS"])]
        name = table.meta["SACCNAME"]
        return NZShiftStretchUncertainty(name, tracer_names, mean, cov)

    def to_table(self):
        """
        Write an NZShiftStretchUncertainty object to an astropy table.

        Parameters
        ----------
        None

        Returns
        -------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.
        """

        data = {
            "ShiftStretch_mean": self.ShiftStretch_mean,
            "ShiftStretch_cov": self.ShiftStretch_cov,
        }
        table = Table(data=data)
        table.meta["N_TRACERS"] = len(self.tracer_names)
        for i, tracer_name in enumerate(self.tracer_names):
            table.meta[f"TRACER_NAME_{i}"] = tracer_name
            table.meta[f"TRACER_NAME_{i}_NPARAMS"] = self.nparams
        return table
