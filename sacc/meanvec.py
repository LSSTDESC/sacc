#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py

class MeanVec(object):
    def __init__ (self, typ, ls, T1, Q1, T2, Q2, value, error, window=None, deltaLS=None, sunit=None):
        self.sunit=sunit ## angular separation unit
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

    def size(self):
        return len(self.data)
                
    def saveToHDF (self, group):
        g=group.create_dataset("mean",data=self.data)
        if self.sunit is not None:
            g.attrs.create("sunit",self.sunit)

    def loadFromHDF (self, dataset):
        raise NotImplementedError
        
