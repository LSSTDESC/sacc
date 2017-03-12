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

    def saveToHDFFile(self,filename):
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()
        
    @classmethod
    def loadFromHDF (Precision, group, mean=None):
        mode=group['precision'].attrs['type']
        data=group['precision'].value
        if mode=="dense":
            matrix=data
        elif mode=="diagonal":
            d=np.diag(data)
        elif mode=="ell_block_diagonal":
            vec=data
            if mean is None:
                print ("Cannot have type==ell_block_diagonal and not mean")
            Np=mean.size()
            matrix=np.zeros((Np,Np))
            k=0
            for i in range(Np):
                for j in range(i,Np):
                    if mean.data["ls"][i]==mean.data["ls"][j]:
                        matrix[i,j]=vec[k]
                        matrix[j,i]=vec[k]
                        k+=1
        return Precision(matrix,mode,mean)
        
        
