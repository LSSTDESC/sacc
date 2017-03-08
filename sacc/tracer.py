#
# SACC : Tracer class
#
from __future__ import print_function, division
import h5py

class Tracer(object):
    def __init__ (self, name=None, Nz=None, exp_sample=None, Nz_sigma_logmean=None,
                  Nz_sigma_logwidth=None, DNz=None):
        self.name=name
        self.Nz=Nz
        self.exp_sample=exp_sample
        self.sigma_logmean=Nz_sigma_logmean
        self.sigma_logwidth=Nz_sigma_logwidth
        self.DNz=Dnz
        self.extra_cols={}
        
    def addColumns (self, columns):
        """
        columns is a dictionary of name, np.array vectors of the same size as Nz.
        """
        self.extra_cols.update(columns)
        for k,c in self.extra_cols.items():
            if (len(c)==len(self.Nz)):
                print("Badly sized column!")
                raise RuntimeError

    def saveToHDF (self, group):
        raise NotImplementedError

    def loadFromHDF (self, dataset):
        raise NotImplementedError
        
