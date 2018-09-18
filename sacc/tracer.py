#
# SACC : Tracer class
#
from __future__ import print_function, division
import h5py
import numpy as np


class Tracer(object):
    def __init__ (self, name, type, z, Nz, exp_sample=None, Nz_sigma_logmean=None,
                  Nz_sigma_logwidth=None, DNz=None, Mproxy_name=None, Mproxy_min=None, Mproxy_max=None):
        self.name=str(name)
        self.type=str(type)
        self.z=z
        self.Nz=Nz
        self.exp_sample=str(exp_sample)
        self.sigma_logmean=Nz_sigma_logmean
        self.sigma_logwidth=Nz_sigma_logwidth
        self.DNz=DNz
        self.Mproxy_name=Mproxy_name#str(Mproxy_name)
        self.Mproxy_min=Mproxy_min
        self.Mproxy_max=Mproxy_max
        self.extra_cols={}

        #For clusters, specify whether the mass proxy information is correctly passed.
        if (self.Mproxy_name is not None):
            print("Name is: "+self.Mproxy_name)
            if self.type != "spin0":
                raise RuntimeError("%s Cluster tracer should have type 'spin0'." % (self.name))
            if (self.Mproxy_min is None) or (self.Mproxy_max is None):
                raise RuntimeError("%s Mproxy_min and Mproxy_max should be specified." % (self.name))
            if (self.Mproxy_min >= self.Mproxy_max):
                raise RuntimeError("%s Mproxy_min should be smaller than Mproxy_max." % (self.name))
        
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

    def is_CL(self):
        if (not(self.Mproxy_name==str(None))) and (self.type == "spin0"):
            return True
        else:
            return False

    def is_WL(self):
        if (self.Mproxy_name==str(None)) and (self.type == "spin2"):
            return True
        else:
            return False
    
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
        print("here:\n",dt)
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
        if self.Mproxy_name is not None:
            a.create("Mproxy_name",self.Mproxy_name.encode('ascii'))
        if self.Mproxy_min is not None:
            a.create("Mproxy_min",self.Mproxy_min)
        if self.Mproxy_max is not None:
            a.create("Mproxy_max",self.Mproxy_max)
        if len(self.extra_cols.keys())>0:
            a.create("extra_cols",[s.encode("ascii") for s in self.extra_cols.keys()])

            
    @classmethod
    def loadFromHDF (Tracer, group, name):
        t=group['tracers'][name]
        d=t.value
        a=t.attrs
        type=str(a['type'])
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
        exp_sample,Nz_sigma_logmean,Nz_sigma_logwidth,Mproxy_name,Mproxy_min,Mproxy_max,ecols=None,None,None,None,None,None,None
        for n,v in a.items():
            if n=='exp_sample': exp_sample=v
            if n=='Nz_sigma_logmean': Nz_sigma_logmean=v
            if n=='Nz_sigma_logwidth': Nz_sigma_logwidth=v
            if n=='Mproxy_name': Mproxy_name=v
            if n=='Mproxy_min': Mproxy_min=v
            if n=='Mproxy_max': Mproxy_max=v
            if n=="extra_cols": ecols=v
        ec={}
        if ecols is not None:
            for n in ecols:
                ec[n.decode()]=d[n.decode()]
        T=Tracer(name,type,z,Nz,exp_sample,Nz_sigma_logmean,Nz_sigma_logwidth,DNz,Mproxy_name,Mproxy_min,Mproxy_max)
        T.addColumns(ec)
        return T
    
