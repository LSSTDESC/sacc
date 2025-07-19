from .base import BaseTracerUncertainty
from ..io import ONE_OBJECT_PER_TABLE


class NZLinearUncertainty(
    BaseTracerUncertainty,
    type_name="nz_Linear"
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
            linear_transformation,
            ):
        """
        Initialize the NZLinearUncertainty object.
        Parameters
        ----------
        tracer_names : list of str
            List of tracer names to which the uncertainty applies.
        mean : array-like
            Intercept of the linear uncertainty model.
            Can be interpreted as the mean of the parameters of the model.
        Linear_transformation : array-like
            Linear transformation matrix of the linear uncertainty model.
            Includes the Cholesky factorization of the covariance matrix
            as well as the linear transformation of the parameters.
        """
        # Check that the mean and linear_transformation have compatible shapes
        assert len(mean) == linear_transformation.shape[0]

        super().__init__(
            name,
            tracer_names,
            mean,
            linear_transformation,
            transformation_type="linear_model"
            )
