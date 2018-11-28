"""
:mod:`sacc` contains one main class and 4 subclasses:

- :class:`sacc.sacc.SACC`
- :class:`sacc.binning.Binning`
- :class:`sacc.tracer.Tracer`
- :class:`sacc.meanvec.MeanVec`
- :class:`sacc.precision.Precision`

Blah
"""

# import individual classes into sacc namespace
from .tracer import Tracer
from .window import Window
from .binning import Binning
from .meanvec import MeanVec
from .precision import Precision
from .sacc import SACC
from .coadd import coadd
