from .base import BaseTracerUncertainty
from ..io import ONE_OBJECT_PER_TABLE
from astropy.table import Table
import numpy as np


class BaseNZGaussianTracerUncertainty(BaseTracerUncertainty):
    # _sub_classes = {}

    def __init__(
            self,
            name,
            tracer_names,
            mean,
            linear_transformation,
            linear_transformation_type):
        """
        Base class for tracer uncertainties that apply to N(z) tracers and act as transforms.

        You don't normally need to instantiate this class directly;
        instead use one of the subclasses.
        
        Parameters
        ----------
        name : str
            Name of the uncertainty.

        tracer_names : list of str
            List of tracer name(s) to which the uncertainty applies.

        mean : array-like
            Mean values for the uncertainty parameters.

        linear_transformation : array-like
            Linear transformation matrix used to generate samples.
            Includes the Cholesky factorization of the covariance matrix
            Might include linear transformation of the parameters.

        transformation_type : str
            Type of transformation used for the uncertainty, currently 
            either "cholesky" or "linear_model".
            Specifies if the transformation is a Cholesky factorization
            or some other type of linear transformation.

            If a 1D array is provided it is interpreted as the diagonal
            of a matrix.
        """
        super().__init__(name, tracer_names)
        self.mean = np.array(mean)
        linear_transformation = np.array(linear_transformation)
        
        # If a 1D array is provided it is interpreted as the diagonal
        if linear_transformation.ndim == 1:
            linear_transformation = np.diag(linear_transformation)
        elif linear_transformation.ndim != 2:
            raise ValueError("linear_transformation must be 1D or 2D array")

        self.linear_transformation = linear_transformation
        self.linear_transformation_type = linear_transformation_type

        # Check that the mean and chol have compatible shapes.
        # Subclasses can do additional shape checks.
        if len(self.mean) != self.linear_transformation.shape[0]:
            raise ValueError(
                f"Mean length {len(self.mean)} does not match "
                f"linear transformation shape {self.linear_transformation.shape}"
            )

        #  if the transformation is a Cholesky factorization
        if linear_transformation_type == "cholesky":
            if self.linear_transformation.shape[0] != self.linear_transformation.shape[1]:
                raise ValueError("Cholesky matrix must be square")

        elif linear_transformation_type == "linear_model":
            # No additional checks needed for linear models
            pass
        else:
            raise ValueError(f"Unknown transformation type: {linear_transformation_type}")

        # Params per tracer
        self.nparams = self.linear_transformation.shape[1] // len(tracer_names)
        

    @classmethod
    def from_table(cls, table):
        """
        Create an instance of one of the NZGaussianTracerUncertainty subclasses
        from an astropy table.

        This method is inherited by all subclasses, so the cls argument will be the
        subclass that calls this method.

        Parameters
        ----------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.

        Returns
        -------
        instance: NZGaussianTracerUncertainty
            An instance of a NZGaussianTracerUncertainty subclass.
        """
        # All the subclasses have the same table structure
        mean = table["mean"]
        transformation_type = table.meta["LINEAR_TRANSFORMATION_TYPE"]
        cholesky = table[transformation_type]
        tracer_names = [table.meta[f"TRACER_NAME_{i}"] for i in range(table.meta["N_TRACERS"])]
        name = table.meta["SACCNAME"]
        return cls(name, tracer_names, mean, cholesky)

    def to_table(self):
        """
        Write an NZGaussianTracerUncertainty object to an astropy table.

        Parameters
        ----------
        None

        Returns
        -------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.
        """

        data = {
            "mean": self.mean,
            self.linear_transformation_type: self.linear_transformation,
        }
        table = Table(data=data)

        # Add metadata - the superclass stores the list of tracer names
        # to which this applies and here we add the rest of the metadata
        self.names_to_table_metadata(table)
        table.meta["LINEAR_TRANSFORMATION_TYPE"] = self.linear_transformation_type
        for i, _ in enumerate(self.tracer_names):
            table.meta[f"TRACER_NAME_{i}_NPARAMS"] = self.nparams
        return table



class NZLinearUncertainty(
    BaseNZGaussianTracerUncertainty,
    type_name="nz_linear_uncertainty",
):
    """
    Class to handle uncertainty in number density tracers represented
    by a linear transformation model.

    In this case to draw a sample of an n(z) you would generate a vector
    of standard normal variables and apply:

    n_i -> n_i + mean + matrix @ normal_vector
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(
            self,
            name,
            tracer_names,
            mean, 
            matrix,
            ):
        """
        Initialize the NZLinearUncertainty object.

        Parameters
        ----------
        tracer_names : list of str
            List of tracer names to which the uncertainty applies.

        mean: array-like
            Intercept of the linear uncertainty model.
            Can be interpreted as the mean of the parameters of the model.

        matrix: array-like
            Linear transformation matrix of the linear uncertainty model.
            Includes the Cholesky factorization of the covariance matrix
            as well as the linear transformation of the parameters.
        """

        super().__init__(
            name,
            tracer_names,
            mean,
            matrix,
            linear_transformation_type="linear_model"
            )


class NZShiftUncertainty(
    BaseNZGaussianTracerUncertainty,
    type_name="nz_shift_uncertainty",
):
    """
    Class to handle uncertainty in number density tracers represented
    by a prior on a translation shift of the redshift distribution.

    In this case to draw a sample of an n(z) you would generate a vector
    of standard normal variables and compute
    n_i(z) -> n_i(z - delta_z)
    where
    delta_z = mean + linear_transformation @ normal_vector
    """
    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(
            self,
            name,
            tracer_names,
            mean,
            cholesky_or_sigma):
        """
        Initialize the NZShiftUncertainty object.
        Parameters
        ----------
        name: str
            Name of the uncertainty object.

        tracer_names : list of str
            List of tracer names to which the uncertainty applies.

        mean: array-like
            Mean shift values for the tracers.

        cholesky_or_sigma: array-like
            cholesky factor of the covariance of the shift values for the tracers.
            You can also supply a 1D array giving the standard deviations
            of the shifts for each tracer.
        """
        super().__init__(
            name,
            tracer_names,
            mean,
            cholesky_or_sigma,
            linear_transformation_type="cholesky"
            )


class NZShiftStretchUncertainty(
    BaseNZGaussianTracerUncertainty,
    type_name="nz_shift_stretch_uncertainty",
):
    """
    Class to handle uncertainty in number density tracers represented
    by a prior on a translation shift of the redshift distribution,
    and on a stretch factor that modifies the width of the distribution.

    In this case to draw a sample of an n(z) you would generate a vector
    of standard normal variables and compute
    n_i(z) -> n_i((z - mean_z) * w + mean_z - delta_z)
    where
    [delta_z_0, delta_z_1, ... delta_z_n, w_0, w_1, .. w_n] = mean + cholesky_or_sigma @ normal_vector
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(
            self,
            name,
            tracer_names,
            mean,
            cholesky_or_sigma):
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
            This is structured as (shift_1, stretch_1, shift_2, stretch_2, ...).
        cholesky_or_sigma : array-like
            Cholesky factor of the covariance of the ShiftStretch 
            values for the tracers. Follows the ordering of the mean.
            Can also be a 1D array giving the standard deviations.
        """

        super().__init__(
            name,
            tracer_names,
            mean,
            cholesky_or_sigma,
            linear_transformation_type="cholesky"
            )
