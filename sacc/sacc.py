#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py



class SACC(object):

    def __init__ (self, tracers, mean=None, precision=None, windows=None):
        self.tracers=tracers
        self.mean=mean
        self.precision=precision
        self.windows=windows
        
    def saveToHDF (self, filename, save_mean=True, save_precision=True):
        f=h5py.File(filename,'w')
        tracer_group=f.create_group("tracers")
        tracer_group.attrs.create("tracer_list",[t.name for t in self.tracers])
        for t in self.tracers:
            t.saveToHDF(tracer_group)
        if self.mean is not None:
            #mean.saveToHDF
            pass
        if self.precision is not None:
            pass
        return 

    @classmethod
    def loadFromHDF (dataset):
        raise NotImplementedError
        
