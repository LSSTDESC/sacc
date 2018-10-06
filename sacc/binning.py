#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py
from .window import Window

def enc(x):
    """
    If an object is a bytes instance or None, return it as-is
    Otherwise if it is a string object, return it as ascii bytes

    Under python 2 bytes and string are the same, so this will
    always return the object as-is.
    """
    if x is None or isinstance(x, bytes):
        return x
    else:
        return x.encode('ascii')

class Binning(object):
    def __init__ (self, typ, ls, T1, Q1, T2, Q2, windows=None, deltaLS=None, sunit=None):
        self.sunit=sunit ## angular separation unit
        self.dtype=[('type','S2'),('ls','f4'), ('T1','i4'),('Q1','S1'), ('T2','i4'),('Q2','S1')]
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


    def cullBinning(self,ndxlist):
        self.binar=self.binar[ndxlist]
        
    def size(self):
        return len(self.binar)

    def get_quantity_pairs(self, typ=None):
        """Return an array of the pairs of quantities included in this data

        Args:
            typ (string, bytes, or None): The type of pair to restrict to. By default use all pairs.

        Returns:
            pairs (array): nx2 array of quantity code pairs (bytes).

        """
        typ = enc(typ)
        quants = [(row['Q1'], row['Q2']) for row in self.binar if (typ==None or row['type']==typ)]
        return np.unique(quants, axis=0)

    def get_bin_pairs(self, Q1, Q2, typ=None):
        """Return an array of the pairs of bin indices included in this data

        Args:
            Q1 (string or bytes): First quantity in pair, e.g. 'S' for shear, 'P' for position
            Q2 (string or bytes): Second quantity in pair
            typ (string, bytes, or None): The type of pair to restrict to. By default use all pairs.
        Returns:
            pairs (array): nx2 array of integer bin pairs.

        """
        Q1 = enc(Q1)
        Q2 = enc(Q2)
        typ = enc(typ)
        pairs = [(row['T1'], row['T2'])
            for row in self.binar
            if row['Q1']==Q1
                and row['Q2']==Q2
                and (typ==None or row['type']==typ)
        ]
        return np.unique(pairs, axis=0)

    def get_angle(self, Q1, Q2, i, j, typ=None):
        """Return an array of the angle (ell or theta) values for a bin

        Args:
            Q1 (string or bytes): First quantity in pair, e.g. 'S' for shear, 'P' for position
            Q2 (string or bytes): Second quantity in pair
            i (int): First bin index
            j (int): Second bin index
            typ (string, bytes, or None): The type of pair to restrict to. By default use all pairs.
        Returns:
            theta (array): Values of the angular quantity. Will be length zero if no matches.

        """        
        Q1 = enc(Q1)
        Q2 = enc(Q2)
        if typ is not None:
            typ = enc(typ)
        theta = [row['ls']
            for row in self.binar
            if row['T1']==i
                and row['T2']==j
                and row['Q1']==Q1
                and row['Q2']==Q2
                and (typ==None or row['type']==typ)
        ]
        return np.array(theta)

                
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
        d['type'] = [typ.decode("utf-8") for typ in d['type']]
        sunit=m.attrs['sunit'] if 'sunit' in m.attrs.keys() else None
        if m.attrs['have_windows']:
            windows=[Window.loadFromHDF(group['windows'],i) for i in range(len(d))]
        else:
            windows=None
        deltaLS=d['Delta_ls'] if 'Delta_ls' in d.dtype.names else None
        return  Binning(d['type'],d['ls'],d['T1'],d['Q1'],d['T2'],d['Q2'],windows=windows, deltaLS=deltaLS, sunit=sunit)
            
        
