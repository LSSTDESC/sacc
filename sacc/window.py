#
# SACC : window class (information for a single window)
#
from __future__ import print_function, division
import h5py
import numpy as np

class Window(object):
    """
    Window objects contain information about the scales that contribute to a given data vector element.
    
    :param array_like ls: array of scales.
    :param array_like w: array of weights for each scale.
    """
    
    def __init__ (self, ls=None, w=None):
        assert(len(ls)==len(w))
        self.ls=ls
        self.w=w/np.sum(w)
        self.nl=len(ls)

    def get_mean_scale(self) :
        """
        Return the mean scale for this window

        :return: weighted mean ell.
        """
        return np.sum(self.w*self.ls)

    def convolve(self,fl) :
        """
        Return the weighed mean of an input array within this window function

        :param array_like fl: input array. Must have the same size as the input ls.
        :return: weighted mean of `fl`
        """
        if len(fl)!=self.nl :
            raise ValueError("Input array must be of size %d"%(self.nl))
        return np.sum(self.w*fl)
    
    def saveToHDF (self, group, ndx):
        """
        Save the windowr to an HDF file.

        :param h5py.Group group: HDF5 group.
        :param int ndx: index of the data vector element that this window corresponds to.
        """
        dtype=[('ls','f4'),('w','f4')]
        ar=np.zeros(len(self.ls),dtype=dtype)
        ar['ls']=self.ls
        ar['w']=self.w
        group.create_dataset("%05d"%ndx,data=ar)

    @classmethod
    def loadFromHDF(Window,group,ndx):
        """
        Create a Window object from an HDF file.

        :param h5py.Group group: HDF5 group where this data vector is saved.
        :param int ndx: index of the data vector element that you want the window function of.
        :return: :class:`Window` object.
        """
        ar=group["%05d"%ndx].value
        return Window(ar['ls'],ar['w'])
