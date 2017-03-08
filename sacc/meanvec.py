#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py

class MeanVec(object):
    def __init__ (self, typ=None, ls=None, T1=None, Q1=None, T2=None, Q2=None, value=None, error=None, window=None, deltaLS=None):
        self.dtype=[('type','S1'),('ls','f4'), ('T1','i4'),('Q1','S1'), ('T2','i4'),('Q2','S1'), ('value','f8'), ('error','f8')]
        if typ is not None:
            N=len(typ)
            if window is not None:
                self.dtype.append(('window','i4'))
            if deltaLS is not None:
                self.dtype.append(('Delta_ls','f4'))
            self.data=np.zeros(N,dtype=self.dtype)
            self.data['type']=typ
            self.data['ls']=ls
            self.data['T1']=T1
            self.data['Q1']=Q1
            self.data['T2']=T2 
            self.data['Q2']=Q2
            self.data['value']=value
            self.data['error']=error
            if window is not None:
                self.data['window']=window
            if deltaLS is not None:
                self.data['Delta_ls']=deltaLS

    def saveToHDF (self, group):
        raise NotImplementedError

    def loadFromHDF (self, dataset):
        raise NotImplementedError
        
