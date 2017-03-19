#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py

class Binning(object):
    def __init__ (self, typ, ls, T1, Q1, T2, Q2, window=None, deltaLS=None, sunit=None):
        self.sunit=sunit ## angular separation unit
        self.dtype=[('type','S1'),('ls','f4'), ('T1','i4'),('Q1','S1'), ('T2','i4'),('Q2','S1')]
        if typ is not None:
            N=len(typ)
            if window is not None:
                self.dtype.append(('window','i4'))
            if deltaLS is not None:
                self.dtype.append(('Delta_ls','f4'))
            self.binar=np.zeros(N,dtype=self.dtype)
            self.binar['type']=typ
            self.binar['ls']=ls
            self.binar['T1']=T1
            self.binar['Q1']=Q1
            self.binar['T2']=T2 
            self.binar['Q2']=Q2
            if window is not None:
                self.binar['window']=window
            if deltaLS is not None:
                self.binar['Delta_ls']=deltaLS

    def size(self):
        return len(self.binar)
                
    def saveToHDF (self, group):
        g=group.create_dataset("binning",data=self.binar)
        if self.sunit is not None:
            g.attrs.create("sunit",self.sunit)

    def saveToHDFFile(self,filename):
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()

    @classmethod
    def loadFromHDF (Binning,group):
        m=group['binning']
        d=m.value
        sunit=m.attrs['sunit'] if 'sunit' in m.attrs.keys() else None
        window=d['window'] if 'window' in d.dtype.names else None
        deltaLS=d['Delta_ls'] if 'Delta_ls' in d.dtype.names else None
        return  Binning(d['type'],d['ls'],d['T1'],d['Q1'],d['T2'],d['Q2'],window, deltaLS,sunit)
            
        
