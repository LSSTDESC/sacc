#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py

class SACC(object):
    def __init__ (self, tracers=None, mean=None, precision=None, windows=None):
        self.tracers=tracers
        self.mean=mean
        self.precision=precision
        self.windows=windows
        
    def saveToHDF (self, group):
        raise NotImplementedError

    def loadFromHDF (self, dataset):
        raise NotImplementedError
        
