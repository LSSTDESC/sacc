#
# SACC : Tracer class
#
from __future__ import print_function, division
import h5py
import numpy as np


class Tracer(object):
    """
    Tracer objects contain information about the maps contributing to any of the 2-point functions stored in a SACC file.
    
    :param str name: Name for the tracer.
    :param str type: The type of tracer, i.e. whether it is a spin0 or spin2 observable. TODO:check this.
    :param str exp_sample: The experiment this tracer corresponds to. While `name` should be unique for each tracer, many tracers can have the same `exp_name`.
    :param array_like z,Nz: arrays describing the redshift distribution of this tracer
    :param float NZ_sigma_logmean: TODO
    :param float NZ_sigma_logwidth: TODO
    :param float DNz: TODO
    :param str Mproxy_name: name of the mass proxy if the tracer is clusters. Defaults to None.
    :param float Mproxy_min: minimum value of the mass proxy bin if the tracer is clusters. Defaults to None.
    :param float Mproxy_max: maximum value of the mass proxy bin if the tracer is clusters. Defaults to None.
    """

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
            #Note, that the "spin0" could be a unicode or byte string.
            #This is a hack until I understand Python 3 strings better...
            if not (self.type != "spin0" or self.type != b"spin0"):
                raise RuntimeError("%s Cluster tracer should have type 'spin0'.\n\tType is %s" % (self.name, self.type))
            if (self.Mproxy_min is None) or (self.Mproxy_max is None):
                raise RuntimeError("%s Mproxy_min and Mproxy_max should be specified." % (self.name))
            if (self.Mproxy_min >= self.Mproxy_max):
                raise RuntimeError("%s Mproxy_min should be smaller than Mproxy_max." % (self.name))
        
    def addColumns (self, columns):
        """
        Adds extra columns describing a tracer (e.g. b(z) or some other function).

        :param dict columns: dictionary where each value should be an array with as many elements as Tracer.Nz. These can be later accessed as `Tracer.extra_cols` or through `Tracer.extraColumn(column_name)`. This function can be called as many times as you want, and this dictionary will be updated with new columns.
        """
        self.extra_cols.update(columns)
        for k,c in self.extra_cols.items():
            if (len(c)!=len(self.Nz)):
                print("Badly sized column!")
                raise RuntimeError

    def extraColumn(self, key):
        """
        Returns a given extra column from this tracer. Literally the same as `Tracer.extra_cols[key]`.

        :param str key: name of the extra column.
        :return: array with the column.
        """
        return self.extra_cols[key]

    def meanZ(self):
        """
        Returns mean redshift for this tracer.

        :return: mean redshift.
        """
        if self.z is None :
            return -1
        return (self.z*self.Nz).sum()/self.Nz.sum()

    def is_CL(self):
        """
        Check if this tracer is a cluster stack.

        :return: `True` or `False`.
        """
        return (self.Mproxy_name is not None) and (self.type == "spin0")

    def is_WL(self):
        """Is the tracer a source for a lensing measurement. Returns a boolean.
        TODO: this doesn't make much sense.
        """
        return (self.Mproxy_name is None) and (self.type == "spin2")
    
    def saveToHDF (self, group):
        """
        Save the tracer to an HDF file.

        TODO: change how cmb is implemented.
        
        :param h5py.Group group: HDF5 group.
        """
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
        #TOM: TODO - wtf is going on here?
        for k,c in self.extra_cols.items():
            # dt.append((k.encode("ascii"),c.dtype))
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
        if self.Mproxy_name is not None:
            a.create("Mproxy_name",self.Mproxy_name.encode('ascii'))
        if self.Mproxy_min is not None:
            a.create("Mproxy_min",self.Mproxy_min)
        if self.Mproxy_max is not None:
            a.create("Mproxy_max",self.Mproxy_max)
        if len(self.extra_cols.keys())>0:
            a.create("extra_cols",[s.encode("ascii") for s in self.extra_cols.keys()])

            
    @classmethod
    def loadFromHDF(Tracer, group, name):
        """
        Create a Tracer object from an HDF file.

        :param h5py.Group group: HDF5 group where this tracer is saved.
        :param str name: name of the tracer to load.
        :return: :class:`Tracer` object.
        """
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
            if n=='Mproxy_name': Mproxy_name=str(v)
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
    
