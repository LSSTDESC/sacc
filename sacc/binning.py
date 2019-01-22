#
# SACC : Binning class
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
    """
    Binning objects contain information about the make up of each element of the data vector stored in a SACC file.
    
    :param array_like typ: array of strings, with each string describing the type of correlation stored in the corresponding element of the data vector. TODO: check this.
    :param array_like ls: array of floats, with each value describing the token scale of the corresponding data vector element.
    :param array_like T1: array of ints, corresponding to the index of the first `Tracer` contributing to this data vector element.
    :param array_like Q1: TODO not sure what this is.
    :param array_like T2: array of ints, corresponding to the index of the second `Tracer` contributing to this data vector element.
    :param array_like Q2: TODO not sure what this is.
    :param array_like windows: array of :class:`sacc.window.Window` objects, describing the contribution of different scales to this element of the data vector. If `None`, delta windows are assumed at the token scales.
    :param array_like deltaLS: TODO not sure what this is.
    :param array_like sunit: array of string, with each string describing the units of the corresponding data vector element.
    """
    
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
        """
        Reduce this Binning object to the indices in `ndxlist`.

        :param array_like ndxlist: list of indices to preserve.
        """
        self.binar=self.binar[ndxlist]
        # Applying binning reduction also to window functions
        if self.windows is not None:
            self.windows = self.windows[ndxlist]
        
    def size(self):
        """
        Returns the size of this Binning object.

        :return: size of the binning object.
        """
        
        return len(self.binar)

    def get_quantity_pairs(self, typ=None):
        """
        Return an array of the pairs of quantities included in this data

        :param str typ: the correlation type to restrict to. If `None` use all pairs.
        :return: array of unique pairs of quantities.
        """
        typ = enc(typ)
        quants = [(row['Q1'], row['Q2']) for row in self.binar if (typ==None or row['type']==typ)]
        return np.unique(quants, axis=0)

    def get_bin_pairs(self, Q1, Q2, typ=None):
        """
        Return an array of the pairs of bin indices included in this data corresponding to the cross-correlation of quantities `Q1` and `Q2` (and additionally of type `typ`).

        TODO: not sure at all why this is a useful function. Also, should rename to `get_tracer_pairs` to keep naming consistent.

        :param str Q1: First quantity in pair, e.g. 'S' for shear, 'P' for position
        :param str Q2: Second quantity in pair.
        :param str typ: The type of pair to restrict to. If None use all pairs.
        :return: nx2 array of tracer index pairs.
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

    def get_angle(self, Q1, Q2, T1, T2, typ=None):
        """
        Return an array with the scale values for a given cross-correlation.

        :param str Q1: First quantity in pair, e.g. 'S' for shear, 'P' for position
        :param str Q2: Second quantity in pair.
        :param str T1: Index of the first tracer.
        :param str T2: Index of the second tracer.
        :param str typ: The type of pair to restrict to. If None use all pairs.
        :return: array of scale values (length zero if no matches found).
        """        
        Q1 = enc(Q1)
        Q2 = enc(Q2)
        if typ is not None:
            typ = enc(typ)
        theta = [row['ls']
            for row in self.binar
            if row['T1']==T1
                and row['T2']==T2
                and row['Q1']==Q1
                and row['Q2']==Q2
                and (typ==None or row['type']==typ)
        ]
        return np.array(theta)

                
    def saveToHDF (self, group):
        """
        Save the binning to an HDF file.

        :param h5py.Group group: HDF5 group.
        """
        g=group.create_dataset("binning",data=self.binar)
        if self.sunit is not None:
            g.attrs.create("sunit",self.sunit)
        g.attrs.create("have_windows",self.windows is not None)
        if self.windows is not None:
            gw=group.create_group("windows")
            for i,w in enumerate(self.windows):
                w.saveToHDF(gw,i)
            
    def saveToHDFFile(self,filename):
        """
        Save the binning to an HDF file.

        :param str filename: path to output file.
        """
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()

    @classmethod
    def loadFromHDF (Binning,group):
        """
        Create a Binning object from an HDF file.

        :param h5py.Group group: HDF5 group where this binning is saved.
        :return: :class:`Binning` object.
        """
        m=group['binning']
        d=m.value
        d['type'] = [typ.decode("utf-8") for typ in d['type']]
        sunit=m.attrs['sunit'] if 'sunit' in m.attrs.keys() else None
        if m.attrs['have_windows']:
            windows=np.array([Window.loadFromHDF(group['windows'],i) for i in range(len(d))])
        else:
            windows=None
        deltaLS=d['Delta_ls'] if 'Delta_ls' in d.dtype.names else None
        return  Binning(d['type'],d['ls'],d['T1'],d['Q1'],d['T2'],d['Q2'],windows=windows, deltaLS=deltaLS, sunit=sunit)
