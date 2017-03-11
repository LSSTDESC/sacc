#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import h5py

class Precision(object):
    def __init__ (self, matrix=None, mode="dense", mean=None):
        ##
        ## mode must be dense, ell_block_diagonal, diagonal
        ##
        if mode not in ["dense", "ell_block_diagonal", "diagonal"]:
            print ("bad precision mode %s", mode)
            raise NotImplemented
        self.matrix=matrix
        self.mode=mode
        self.mean=mean
        
    def saveToHDF (self, group): ## might need mean for certain modes of saving
        if self.mode=="dense":
            d=group.create_dataset("precision",data=self.matrix)
        elif self.mode=="diagonal":
            d=group.create_dataset("precision", data=self.matrix.diagonal())
        elif self.mode=="ell_block_diagonal":
            vec=[]
            Np=self.mean.size()
            for i in range(Np):
                for j in range(i,Np):
                    if self.mean.data["ls"][i]==self.mean.data["ls"][j]:
                        vec.append(self.matrix[i,j])
            d=group.create_dataset("precision",data=vec)
        d.attrs.create("type",self.mode)

    def loadFromHDF (self, dataset, mean=None):
        raise NotImplementedError
        
