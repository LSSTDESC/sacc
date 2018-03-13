#
# SACC : window class (information for a single window)
#
from __future__ import print_function, division
import h5py
import numpy as np

class Window(object):
    def __init__ (self, ls=None, w=None):
        assert(len(ls)==len(w))
        self.ls=ls
        self.w=w

    def saveToHDF (self, group, ndx):
        dtype=[('ls','f4'),('w','f4')]
        ar=np.zeros(len(self.ls),dtype=dtype)
        ar['ls']=self.ls
        ar['w']=self.w
        group.create_dataset("%05d"%ndx,data=ar)

    @classmethod
    def loadFromHDF(Window,group,ndx):
        ar=group["%05d"%ndx].value
        return Window(ar['ls'],ar['w'])
    
