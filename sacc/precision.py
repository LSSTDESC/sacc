#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py

class Precision(object):
    def __init__ (self, matrix=None):
        self.matrix=matrix

    def saveToHDF (self, group, mode="dense", mean=None): ## might need mean for certain modes of saving
        raise NotImplementedError

    def loadFromHDF (self, dataset, mean=None):
        raise NotImplementedError
        
