from .base import BaseTracerUncertainty
from ..io import ONE_OBJECT_PER_TABLE
from astropy.table import Table
import numpy as np


class NZLinearUncertainty(BaseTracerUncertainty, type_name="nz_Linear"):
    """
    Class to handle uncertainty in number density tracers.
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, name, tracer_names, Linear_intercept, Linear_slope):
        """
        Initialize the NZLinearUncertainty object.
        Parameters
        ----------
        tracer_names : list of str
            List of tracer names to which the uncertainty applies.
        Linear_intercept : array-like
            Intercept of the linear uncertainty model.
            Can be interpreted as the mean of the parameters of the model.
        Linear_slope : array-like
            Slope of the linear uncertainty model.
            If square, it can be interpreted as the covariance of
            the parameters of the model.
            If not, it is interpreted as the linear transformation matrix
            which already includes the covariance of the parameters.
        """

        super().__init__(name, tracer_names)
        self.Linear_intercept = np.array(Linear_intercept)
        self.Linear_slope = np.array(Linear_slope)
        if type(tracer_names) == str:
            self.nparams = len(self.Linear_intercept)
        else:
            self.nparams = len(self.Linear_intercept) // len(tracer_names)
        n, m = self.Linear_slope.shape
        if n != m:
            self.slope_type = "linear_model_matrix"
        else:
            self.slope_type = "covariance_matrix"

    @classmethod
    def from_table(self, table):
        """
        Read an NZLinearUncertainty object from an astropy table.

        Parameters
        ----------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.

        Returns
        -------
        instance: NZLinearUncertainty
            An instance of the NZLinearUncertainty class.
        """
        mean = table["Linear_mean"]
        cov = table["Linear_cov"]
        tracer_names = [table.meta[f"TRACER_NAME_{i}"] for i in range(table.meta["N_TRACERS"])]
        name = table.meta["SACCNAME"]
        return NZLinearUncertainty(name, tracer_names, mean, cov)

    def to_table(self):
        """
        Write an NZLinearUncertainty object to an astropy table.

        Parameters
        ----------
        None

        Returns
        -------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.
        """

        data = {
            "Linear_intercept": self.Linear_intercept,
            "Linear_slope": self.Linear_slope,
        }
        table = Table(data=data)
        table.meta["N_TRACERS"] = len(self.tracer_names)
        table.meta["slope_type"] = self.slope_type
        for i, tracer_name in enumerate(self.tracer_names):
            table.meta[f"TRACER_NAME_{i}"] = tracer_name
            table.meta[f"TRACER_Name_{i}_NPARAMS"] = self.nparams
        return table
