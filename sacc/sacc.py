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
        if windows is not None:
            print ("Windows not yet implemented. Will be ignored.")
        
    def saveToHDF (self, filename, save_mean=True, save_precision=True, mean_filename=None, precision_filename=None):
        f=h5py.File(filename,'w')
        tracer_group=f.create_group("tracers")
        tracer_group.attrs.create("tracer_list",[t.name for t in self.tracers])
        for t in self.tracers:
            t.saveToHDF(tracer_group)
        if save_mean:
            if self.mean is not None:
                self.mean.saveToHDF(f)
        else:
            if mean_filename is not None:
                f.attrs['mean_file_path']=mean_filename
        if self.precision is not None:
            self.precision.saveToHDF(f)
        return 

    @classmethod
    def loadFromHDF (filename,mean_filename=None, precision_filename=None):
        f=h5py.File(filename,'r')
        tracer_group=f['tracers']
        tnames=tracer_group.attrs['tracer_list']
        tracers=[Tracer.loadFromHDF(n) for n in tnames]
        ##
        ## if mean specified, we use it, otherwise look for mean group and mean filename
        ## in attributes.
        ##
        
