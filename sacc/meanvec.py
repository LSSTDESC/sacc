#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py

class MeanVec(object):
    def __init__ (self, values):
        self.data=values

    def size(self):
        return len(self.data)
                
    def saveToHDF (self, group):
        g=group.create_dataset("mean",data=self.data)

    def saveToHDFFile(self,filename):
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()

    @classmethod
    def loadFromHDF (MeanVec,group):
        m=group['mean']
        d=m.value
        return  MeanVec(d)
    
            
        
