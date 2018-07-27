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
            if (len(c)!=len(self.Nz)):
                print("Badly sized column!")
                raise RuntimeError

    def extraColumn(self, key):
        return self.extra_cols[key]

    def meanZ(self):
        if self.z is None :
            return -1
        return (self.z*self.Nz).sum()/self.Nz.sum()
            
    def saveToHDF (self, group):
        ## if CMB, go empty
        if self.type=="cmb":
            g=group.create_dataset(self.name,data=[])
            g.attrs['type']=self.type
            return 

        ## first create dataset
        dt=[('z',np.dtype('f4')),('Nz',np.dtype('f4'))]
        if self.z is None :
            lenz=1
        else :
            lenz=len(self.z)

        if self.DNz is not None:
            _,numDNz=self.DNz.shape
        else:
            numDNz=0
        for i in range(numDNz):
            dt.append(("DNz_"+str(i),'f4'))
        for k,c in self.extra_cols.items():
            #dt.append((k.encode("ascii"),c.dtype))
            dt.append((k,c.dtype))
            #dt.append(("b",c.dtype))
        data=np.zeros(lenz,dtype=dt)
        if self.z is not None :
            data['z']=self.z
        else :
            data['z']=np.array([-1.])
        if self.Nz is not None :
            data['Nz']=self.Nz
        else :
            data['Nz']=np.array([-1.])
        for i in range(numDNz):
            data['DNz_'+str(i)]=self.DNz[:,i]
        for k,c in self.extra_cols.items():
            data[k]=c
        dset=group.create_dataset(self.name, data=data)
        a=dset.attrs
        a.create("type",self.type.encode('ascii'))
        if self.exp_sample is not None:
            a.create("exp_sample",self.exp_sample.encode('ascii'))
        if self.sigma_logmean is not None:
            a.create("Nz_sigma_logmean",self.sigma_logmean)
        if self.sigma_logwidth is not None:
            a.create("Nz_sigma_logwidth",self.sigma_logwidth)
        if len(self.extra_cols.keys())>0:
            a.create("extra_cols",[s.encode("ascii") for s in self.extra_cols.keys()])

            
    @classmethod
    def loadFromHDF (Tracer, group, name):
        t=group['tracers'][name]
        d=t.value
        a=t.attrs
        type=a['type']
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
        if ecols is not None:
            for n in ecols:
                ec[n.decode()]=d[n.decode()]
        T=Tracer(name,type,z,Nz,exp_sample,Nz_sigma_logmean,Nz_sigma_logwidth,DNz)
        T.addColumns(ec)
        return T
    
