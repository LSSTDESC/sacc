from ..io import BaseIO


class BaseTracerUncertainty(BaseIO):
    _sub_classes = {}

    def __init__(
            self,
            name,
            tracer_names,
            mean,
            linear_transformation,
            linear_transformation_type="chol"):
        """
        Base class for tracer uncertainties.  All uncertainty subclasses
        must define a list of tracer names to which the uncertainty applies.

        This could be a single tracer or multiple tracers.

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
            Type of transformation used for the uncertainty.
            Specifies if the transformation is a Cholesky factorization
            or some other type of linear transformation.
        """
        super().__init__()
        self.name = name
        self.tracer_names = tracer_names
        self.mean = mean
        self.linear_transformation = linear_transformation
        self.linear_transformation_type = linear_transformation_type
        # Check that the mean and chol have compatible shapes
        assert len(self.mean) == linear_transformation.shape[0]
        #  if the transformation is a Cholesky factorization
        if linear_transformation_type == "chol":
            assert linear_transformation.shape[0] == linear_transformation.shape[1]
        elif linear_transformation_type == "linear_model":
            pass  # No additional checks needed for linear models
        else:
            raise ValueError(f"Unknown transformation type: {linear_transformation_type}")
        # Params per tracer
        self.nparams = linear_transformation.shape[1] // len(tracer_names)

    @classmethod
    def from_table(self, table):
        """
        Create an NZShiftStretchUncertainty object from an astropy table.

        Parameters
        ----------
        table: astropy.table.Table
            An astropy table containing the uncertainty data.

        Returns
        -------
        instance: BaseTracerUncertainty
            An instance of the BaseTracerUncertainty class.
        """
        mean = table["mean"]
        chol = table[self.linear_transformation_type]
        tracer_names = [table.meta[f"TRACER_NAME_{i}"] for i in range(table.meta["N_TRACERS"])]
        name = table.meta["SACCNAME"]
        return self(name, tracer_names, mean, chol)

    def to_table(self, table):
        """
        Write an BaseTracerUncertainty object to an astropy table.

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
        table = table(data=data)
        table.meta["N_TRACERS"] = len(self.tracer_names)
        for i, tracer_name in enumerate(self.tracer_names):
            table.meta[f"TRACER_NAME_{i}"] = tracer_name
            table.meta[f"TRACER_NAME_{i}_NPARAMS"] = self.nparams
        return table
