from .base import BaseTracerUncertainty
from ..io import ONE_OBJECT_PER_TABLE


class NZShiftUncertainty(
    BaseTracerUncertainty,
    type_name="nz_shift",
):
    """
    Class to handle uncertainty in number density tracers.
    """
    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(
            self,
            name,
            tracer_names,
            mean,
            chol):
        """
        Initialize the NZShiftUncertainty object.
        Parameters
        ----------
        name : str
            Name of the uncertainty object.
        tracer_names : list of str
            List of tracer names to which the uncertainty applies.
        mean : array-like
            Mean shift values for the tracers.
        chol : array-like
            choleskey factor of the covariance of the shift values for the tracers.
        """
        super().__init__(
            name,
            tracer_names,
            mean,
            chol,
            transformation_type="chol"
            )
