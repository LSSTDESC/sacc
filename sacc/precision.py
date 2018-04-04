#
# SACC : Tracer class
#
from __future__ import print_function, division
import numpy as np
import scipy.linalg as la
import h5py

class Precision(object):
    def __init__ (self, matrix=None, mode="dense", is_covariance=False, binning=None):
        ##
        ## mode must be dense, ell_block_diagonal, diagonal
        ##
        ## If is_covariance is True, we're passing covariance matrix rather than 
        ## precision
        if mode not in ["dense", "ell_block_diagonal", "diagonal"]:
            print ("bad precision mode %s", mode)
            raise NotImplemented

        if is_covariance:
            self.cmatrix=matrix
            self.pmatrix=None
        else:
            self.pmatrix=matrix
            self.cmatrix=None
            
        self.mode=mode
        self.binning=binning
        

    def CullMatrix(self,ndxlist):
        if self.cmatrix is not None:
            _getCovarianceFromPrecision()
        if self.mode=="diagonal":
            self.cmatrix=self.cmatrix(ndxlist)
        elif (self.mode in ['dense','ell_block_diagonal']):
            N=len(ndxlist)
            cmatrix=np.zeros((N,N))
            ## there should be a better way of doing this:
            for i in range(N):
                cmatrix[i,ndx]=self.cmatrix[ndx[i],ndx]
            self.cmatrix=cmatrix

        if self.pmatrix is not None:
            _getPrecisionFromCovariance()
        
    def _getCovarianceFromPrecision(self):
        if self.pmatrix is None:
            print ("Consider getting a job in McDonalds.")
            raise AssertionErrror()
        if self.mode=='diagonal':
            self.cmatrix=1/la.inv(self.pmatrix)
        elif (self.mode in ['dense','ell_block_diagonal']):
            self.cmatrix=la.inv(self.pmatrix)
        else:
            raise NotImplementedError()
        
    def _getPrecisionFromCovariance(self):
        if self.cmatrix is None:
            print ("Consider getting a job in McDonalds.")
            raise AssertionErrror()
        if self.mode=='diagonal':
            self.pmatrix=1/la.inv(self.cmatrix)
        elif (self.mode in ['dense','ell_block_diagonal']):
            self.pmatrix=la.inv(self.cmatrix)
        else:
            raise NotImplementedError()

    def precisionMatrix(self):
        if self.pmatrix is None:
            self._getPrecisionFromCovariance()
        return self.pmatrix
    
            
    def saveToHDF (self, group): ## might need binning for certain modes of saving
        ## if we have covariance matrix, save that one, otherwise save precision
        if self.cmatrix is not None:
            matrix=self.cmatrix
            savingC=True
        else:
            matrix=self.pmatrix
            savingC=False

        if self.mode=="dense":
            d=group.create_dataset("error",data=matrix)
        elif self.mode=="diagonal":
            d=group.create_dataset("error", data=matrix.diagonal())
        elif self.mode=="ell_block_diagonal":
            vec=[]
            Np=self.binning.size()
            for i in range(Np):
                for j in range(i,Np):
                    if self.binning.binar["ls"][i]==self.binning.binar["ls"][j]:
                        vec.append(matrix[i,j])
            d=group.create_dataset("error",data=vec)
        d.attrs.create("type",self.mode.encode("ascii"))
        d.attrs.create("is_covariance",savingC)

        
    def saveToHDFFile(self,filename):
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()
        
    @classmethod
    def loadFromHDF (Precision, group, binning=None):
        D=group['error']
        mode=D.attrs['type'].decode()
        loadingC=D.attrs['is_covariance']
        data=group['error'].value
        if mode=="dense":
            matrix=data
        elif mode=="diagonal":
            matrix=np.diag(data)
        elif mode=="ell_block_diagonal":
            vec=data
            if binning is None:
                print ("Cannot have type==ell_block_diagonal and not binning")
            Np=binning.size()
            matrix=np.zeros((Np,Np))
            k=0
            for i in range(Np):
                for j in range(i,Np):
                    if binning.binar["ls"][i]==binning.binar["ls"][j]:
                        matrix[i,j]=vec[k]
                        matrix[j,i]=vec[k]
                        k+=1
        return Precision(matrix,mode,is_covariance=loadingC,binning=binning)
        
        
