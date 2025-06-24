from ..io import BaseIO

class BaseTracerUncertainty(BaseIO):
    _sub_classes = {}

    def __init__(self, name, tracer_names):
        """
        Base class for tracer uncertainties.  All uncertainty subclasses
        must define a list of tracer names to which the uncertainty applies.

        This could be a single tracer or multiple tracers.
        
        Parameters
        ----------
        tracer_names : list of str
            List of tracer name(s) to which the uncertainty applies.
        """
        super().__init__()
        self.name = name
        self.tracer_names = tracer_names
