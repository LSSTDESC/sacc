#
# SACC : window class (information for a single window)
#
from __future__ import print_function, division
import h5py

class Window(object):
    def __init__ (self, ls=None, w=None, units=None):
        self.ls=ls
        self.w=w
        self.units=units

    def saveToHDF (self, group, name):
        raise NotImplementedError
    
    def loadFromHDF(self, dataset):
        raise NotImplementedError
