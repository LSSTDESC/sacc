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
        if (self.type=="cmb"):
            self.exp_sample=name
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
            if (len(c)!=len(self.Nz)):
                print("Badly sized column!")
                raise RuntimeError

    def extraColumn(self, key):
        return self.extra_cols[key]

    def meanZ(self):
        if self.type=="cmb":
            return -1 ## yes, yes, we could be funny and put 1150 here.
        return (self.z*self.Nz).sum()/self.Nz.sum()
            
    def saveToHDF (self, group):
        ## first create dataset
        if self.type=="cmb":
            g=group.create_dataset(self.name,data=[])
            g.attrs['type']=self.type
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
        if len(self.extra_cols.keys())>0:
            a.create("extra_cols",self.extra_cols.keys())
            
    @classmethod
    def loadFromHDF (Tracer, group, name):
        t=group['tracers'][name]
        d=t.value
        a=t.attrs
        type=a['type']
        if type=='cmb':
            return Tracer(name,type,[],[])
        z=d['z']
        Nz=d['Nz']
        numDNZ=0
        DNZ=[]
        while True:
            fn="DNz_"+str(DNZ)
            if fn in d.dtype.names:
                DNZ.append(d[fn])
                DNZ+=1
            else:
                break
        if numDNZ==0:
            DNz=None
        else:
            DNz=np.array(DNZ).T
        exp_sample,Nz_sigma_logmean,Nz_sigma_logwidth,ecols=None,None,None,None
        for n,v in a.items():
            if n=='exp_sample': exp_sample=v
            if n=='Nz_sigma_logmean': Nz_sigma_logmean=v
            if n=='Nz_sigma_logwidth': Nz_sigma_logwidth=v
            if n=="extra_cols": ecols=v
        ec={}
        for n in ecols:
            ec[n]=d[n]
        T=Tracer(name,type,z,Nz,exp_sample,Nz_sigma_logmean,Nz_sigma_logwidth,DNz)
        T.addColumns(ec)
        return T
    
