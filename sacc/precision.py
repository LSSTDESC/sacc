#
# SACC : Precision class
#
from __future__ import print_function, division
import numpy as np
import scipy.linalg as la
import h5py

class Precision(object):
    """
    Precision objects contain the covariance (or inverse covariance a.k.a. precision) matrix for the data vector stored in a SACC file.
    
    :param array_like matrix: covariance or precision matrix.
    :param str mode: how is the matrix passed? Options: `dense` (full matrix), `diagonal` (only diagonal passed, assume off-diagonal is zero), `ell_block_diagonal` (matrix is diagonal between different scales).
    :param boolean is_covariance: set to True (default) if passing the covariance matrix. Otherwise, `matrix` should contain the precision matrix.
    :param Binning binning: a Binning object describing the data vector of this covariance. Only needed if `mode=='ell_block_diagonal'`.
    """

    def __init__ (self, matrix=None, mode="dense", is_covariance=True, binning=None):
        ##
        ## mode must be dense, ell_block_diagonal, diagonal
        ##
        ## If is_covariance is True, we're passing covariance matrix rather than 
        ## precision
        if mode not in ["dense", "ell_block_diagonal", "diagonal"]:
            print ("bad precision mode %s", mode)
            raise NotImplemented

        if is_covariance:
            self._cmatrix=matrix
            self._pmatrix=None
        else:
            self._pmatrix=matrix
            self._cmatrix=None
            
        self.mode=mode
        self.binning=binning
        

    def cullMatrix(self,ndxlist):
        """
        Reduce this Precision object to the indices in `ndxlist`.

        :param array_like ndxlist: list of indices to preserve.
        """
        if self._cmatrix is None:
            self._getCovarianceFromPrecision()
        if self.mode=="diagonal":
            self._cmatrix=self._cmatrix(ndxlist)
        elif (self.mode in ['dense','ell_block_diagonal']):
            N=len(ndxlist)
            cmatrix=np.zeros((N,N))
            ## there should be a better way of doing this:
            for i in range(N):
                cmatrix[i,:]=self._cmatrix[ndxlist[i],ndxlist]
            self._cmatrix=cmatrix

        if self._pmatrix is not None:
            self._getPrecisionFromCovariance()
        
    def _getCovarianceFromPrecision(self):
        if self._pmatrix is None:
            print ("Consider getting a job in McDonalds.")
            raise AssertionErrror()
        if self.mode=='diagonal':
            self._cmatrix=1/la.inv(self._pmatrix)
        elif (self.mode in ['dense','ell_block_diagonal']):
            self._cmatrix=la.inv(self._pmatrix)
        else:
            raise NotImplementedError()
        
    def _getPrecisionFromCovariance(self):
        if self._cmatrix is None:
            print ("Consider getting a job in McDonalds.")
            raise AssertionErrror()
        if self.mode=='diagonal':
            self._pmatrix=1/la.inv(self._cmatrix)
        elif (self.mode in ['dense','ell_block_diagonal']):
            self._pmatrix=la.inv(self._cmatrix)
        else:
            raise NotImplementedError()

    def getPrecisionMatrix(self):
        """
        Get precision matrix stored in this object.

        :return: precision matrix (inverse covariance).
        """
        if self._pmatrix is None:
            self._getPrecisionFromCovariance()
        return self._pmatrix

    def getCovarianceMatrix(self):
        """
        Get covariance matrix stored in this object.

        :return: covariance matrix.
        """
        if self._cmatrix is None:
            self._getCovarianceFromPrecision()
        return self._cmatrix
            
    def saveToHDF (self, group):
        """
        Save the precision object to an HDF file. If the covariance matrix exists then that one is saved, otherwise the precision is saved.

        :param h5py.Group group: HDF5 group.
        """
        ## might need binning for certain modes of saving
        ## if we have covariance matrix, save that one, otherwise save precision
        if self._cmatrix is not None:
            matrix=self._cmatrix
            savingC=True
        else:
            matrix=self._pmatrix
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
        """
        Save the precision object to an HDF file.

        :param str filename: path to output file.
        """
        f=h5py.File(filename,'w')
        self.saveToHDF(f)
        f.close()
        
    @classmethod
    def loadFromHDF (Precision, group, binning=None):
        """
        Create a Precision object from an HDF file.

        :param h5py.Group group: HDF5 group where this precision object is saved.
        :param Binning binning: a Binning object describing the data vector of this covariance. Only needed if `mode=='ell_block_diagonal'`.
        :return: :class:`Precision` object.
        """
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
