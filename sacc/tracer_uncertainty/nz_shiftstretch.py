from .base import BaseTracerUncertainty
from ..io import ONE_OBJECT_PER_TABLE


class NZShiftStretchUncertainty(
    BaseTracerUncertainty,
    type_name="nz_ShiftStretch",
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
        Initialize the NZShiftStretchUncertainty object.
        Parameters
        ----------
        name : str
            Name of the uncertainty object.
        tracer_names : list of str
            List of tracer names to which the uncertainty applies.
        mean : array-like
            Mean ShiftStretch values for the tracers.
            This is strucutred as (shift_1, stretch_1, shift_2, stretch_2, ...).
        chol : array-like
            Cholesky factor of the covariance of the ShiftStretch 
            values for the tracers. Follows the ordering of the mean
        """

        super().__init__(
            name,
            tracer_names,
            mean,
            chol,
            transformation_type="chol"
            )
