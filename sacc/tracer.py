#
# SACC : Tracer class
#
from __future__ import print_function, division
import h5py
import numpy as np


class Tracer(object):
    def __init__ (self, name, type, z, Nz, exp_sample=None, Nz_sigma_logmean=None,
                  Nz_sigma_logwidth=None, DNz=None):
        self.name=name
        self.type=type
        self.z=z
        self.Nz=Nz
        self.exp_sample=exp_sample
        self.sigma_logmean=Nz_sigma_logmean
        self.sigma_logwidth=Nz_sigma_logwidth
        self.DNz=DNz
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
        ## first create dataset
        if self.type=="cmb":
            group.create_dataset(self.name,data=[])
            return 
        dt=[('z','f4'), ('Nz','f4')]
        if self.DNz is not None:
            _,numDNz=self.DNz.shape
        else:
            numDNz=0
        for i in range(numDNz):
            dt.append(("DNz_"+str(i),'f4'))
        for k,c in self.extra_cols.items():
            dt.append((k,c.dtype))
        data=np.zeros(len(self.Nz),dtype=dt)
        data['z']=self.z
        data['Nz']=self.Nz
        for i in range(numDNz):
            data['DNz_'+str(i)]=self.DNz[:,i]
        for k,c in self.extra_cols.items():
            data[k]=c
        dset=group.create_dataset(self.name, data=data)
        a=dset.attrs
        a.create("type",self.type)
        if self.exp_sample is not None:
            a.create("exp_sample",self.exp_sample)
        if self.sigma_logmean is not None:
            a.create("Nz_sigma_logmean",self.sigma_logmean)
        if self.sigma_logwidth is not None:
            a.create("Nz_sigma_logwidth",self.sigma_logwidth)
        
    def loadFromHDF (self, dataset):
        raise NotImplementedError
        
