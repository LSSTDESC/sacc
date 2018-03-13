#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py
from window import Window

class Binning(object):
    def __init__ (self, typ, ls, T1, Q1, T2, Q2, windows=None, deltaLS=None, sunit=None):
        self.sunit=sunit ## angular separation unit
        self.dtype=[('type','S1'),('ls','f4'), ('T1','i4'),('Q1','S1'), ('T2','i4'),('Q2','S1')]
        if typ is not None:
            N=len(typ)
            self.windows=windows
            if deltaLS is not None:
                self.dtype.append(('Delta_ls','f4'))
            self.binar=np.zeros(N,dtype=self.dtype)
            self.binar['type']=typ
            self.binar['ls']=ls
            self.binar['T1']=T1
            self.binar['Q1']=Q1
            self.binar['T2']=T2 
            self.binar['Q2']=Q2
            if deltaLS is not None:
                self.binar['Delta_ls']=deltaLS

    def size(self):
        return len(self.binar)
                
    def saveToHDF (self, group):
        g=group.create_dataset("binning",data=self.binar)
        if self.sunit is not None:
            g.attrs.create("sunit",self.sunit)
        g.attrs.create("have_windows",self.windows is not None)
        if self.windows is not None:
            gw=group.create_group("windows")
            for i,w in enumerate(self.windows):
                w.saveToHDF(gw,i)
            
    def saveToHDFFile(self,filename):
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()

    @classmethod
    def loadFromHDF (Binning,group):
        m=group['binning']
        d=m.value
        sunit=m.attrs['sunit'] if 'sunit' in m.attrs.keys() else None
        if m.attrs['have_windows']:
            windows=[Window.loadFromHDF(group['windows'],i) for i in range(len(d))]
        else:
            windows=None
        deltaLS=d['Delta_ls'] if 'Delta_ls' in d.dtype.names else None
        return  Binning(d['type'],d['ls'],d['T1'],d['Q1'],d['T2'],d['Q2'],windows=windows, deltaLS=deltaLS, sunit=sunit)
            
        
