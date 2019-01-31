#
# SACC : MeanVec class
#
from __future__ import print_function, division
import numpy as np
import h5py

class MeanVec(object):
    """
    MeanVec objects contain the data vector stored in a SACC file.
    
    :param array_like values: data vector.
    """

    def __init__ (self, values):
        self.vector=values

    def size(self):
        """
        Returns the size of the data vector
        
        :return: data vector size.
        """
        return len(self.vector)

    def cullVector(self,ndxlist):
        """
        Reduce this MeanVec object to the indices in `ndxlist`.

        :param array_like ndxlist: list of indices to preserve.
        """
        self.vector=self.vector[ndxlist]
    
    def saveToHDF (self, group):
        """
        Save the data vector to an HDF file.

        :param h5py.Group group: HDF5 group.
        """
        g=group.create_dataset("mean",data=self.vector)

    def saveToHDFFile(self,filename):
        """
        Save the data vector to an HDF file.

        :param str filename: path to output file.
        """
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()

    @classmethod
    def loadFromHDF (MeanVec,group):
        """
        Create a MeanVec object from an HDF file.

        :param h5py.Group group: HDF5 group where this data vector is saved.
        :return: :class:`MeanVec` object.
        """
        m=group['mean']
        d=m.value
        return  MeanVec(d)
    
            
        
