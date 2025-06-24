from .base import BaseTracerUncertainty
from ..io import ONE_OBJECT_PER_TABLE
from astropy.table import Table
import numpy as np

class NZShiftUncertainty(BaseTracerUncertainty, type_name="nz_shift"):
    """
    Class to handle uncertainty in number density tracers.
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, name, tracer_names, shift_mean, shift_sigma):
        """
        Initialize the NZShiftUncertainty object.
        Parameters
        ----------
        tracer_names : list of str
            List of tracer names to which the uncertainty applies.
        shift_mean : array-like
            Mean shift values for the tracers.
        shift_sigma : array-like
            Standard deviation of the shift values for the tracers.
        """

        super().__init__(name, tracer_names)
        self.shift_mean = np.array(shift_mean)
        self.shift_sigma = np.array(shift_sigma)

    @classmethod
    def from_table(self, table):
        """
        Read an NZShiftUncertainty object from an astropy table.

        Parameters
        ----------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.

        Returns
        -------
        instance: NZShiftUncertainty
            An instance of the NZShiftUncertainty class.
        """
        mean = table["shift_mean"]
        sigma = table["shift_sigma"]
        tracer_names = [table.meta[f"TRACER_NAME_{i}"] for i in range(table.meta["N_TRACERS"])]
        name = table.meta["SACCNAME"]
        return NZShiftUncertainty(name, tracer_names, mean, sigma)
    
    def to_table(self):
        """
        Write an NZShiftUncertainty object to an astropy table.

        Parameters
        ----------
        None

        Returns
        -------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.
        """

        data = {
            "shift_mean": self.shift_mean,
            "shift_sigma": self.shift_sigma,
        }
        table = Table(data=data)
        table.meta["N_TRACERS"] = len(self.tracer_names)
        for i, tracer_name in enumerate(self.tracer_names):
            table.meta[f"TRACER_NAME_{i}"] = tracer_name
        return table    

