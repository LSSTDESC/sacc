"""
SACC is the main container class of the SACC module.

"""


from __future__ import print_function, division
from .tracer import Tracer
from .binning import Binning
from .meanvec import MeanVec
from .precision import Precision
import numpy as np
import h5py
import sys
PY3 = sys.version_info[0] == 3


class SACC(object):
    
    ## format version, bump up every time you break thins
    ## irreparrably
    _format_version=1


    """
    The main container class for 2pt measurements.

    Parameters
    ----------

    tracers : list of Tracer objects
       List of tracer objects used in the measurement and referenced in 
       binning parameter
    binning : Binning object
       Binning object describing what measurement does each index in the mean
       vector and precision matrix contain
    mean : array_like or MeanVec object
       Vector representing the actual measurement, the mean of the gaussian.
       If not a MeanVec object, will try to cast it into one 
    precision: Precision object
       Object representing the precision (i.e. inverse covariance) matrix.

    """

    
    def __init__ (self, tracers, binning, mean=None, precision=None, meta={}):
        self.tracers=tracers
        self.binning=binning
        assert([type(t)==type(Tracer) for t in self.tracers])
        #print(type(self.binning),type(Binning))
        #assert(type(self.binning)==type(Binning)) ## this assert always fails, not sure why
        if type(mean)!=MeanVec:
            mean=MeanVec(mean)
        self.mean=mean
        self.precision=precision
        self.meta=meta
        
    def get_exp_sample_set(self):
        return set([t.exp_sample for t in self.tracers])
        
        
    def printInfo(self):
        exps=self.get_exp_sample_set()
        print ("--------------------------------")

        if len(self.meta)>0:
            print ("Meta information:")
            for key,value in (self.meta.items() if PY3 else self.meta.iteritems()):
                print ("  ",key,":",value)
            print ("--------------------------------")
        for e in exps:
            print (" EXP_SAMPLE:",e)
            cc=0
            for t in self.tracers:
                if t.exp_sample==e:
                    print ("       Tomographic sample #%i: <z>=%4.2f"%(cc,t.meanZ()))
                    cc+=1
        print ("--------------------------------")
        if self.binning is not None:
            print ("Total number of points:",self.binning.size())
        else:
            print ("No mean vector.")
        if self.precision is not None:
            print ("Precision matrix type:",self.precision.mode)
        else:
            print ("No precision matrix.")


    def size(self):
        return self.binning.size()

    def lrange(self,t1i,t2i):
        ndx=np.where( ((self.binning.binar['T1']==t1i) & (self.binning.binar['T2']==t2i)) |
                      ((self.binning.binar['T1']==t2i) & (self.binning.binar['T2']==t1i)) )
        return self.binning.binar['ls'][ndx]

    def ilrange(self,t1i,t2i):
        ndx=np.where( ((self.binning.binar['T1']==t1i) & (self.binning.binar['T2']==t2i)) |
                      ((self.binning.binar['T1']==t2i) & (self.binning.binar['T2']==t1i)) )
        return ndx

        
    def sortTracers(self):
        """Sort the tracers and return information on them in a tuple form.
        The return format is: (t1i,t2i,type,ells,ndx),
        where t1i,t2i are indices of first and second tracer,
        type is the kind of data vector,
        ells is the list of ells used, and ndx is the list of indices in the
        data vector corresponding to this.

        TODO: figure out how to make this compatible with 3-pt correlation functions.

        Returns:
            (tuple): Information about all tracers and how they are combined
        """
        Nt=len(self.tracers)
        toret=[]
        #Loop over all tracers
        for t1i in range(Nt):
            #Check if this tracer is a number count, '+N' type
            n_cl   = np.where( ((self.binning.binar['T1']==t1i) & (self.binning.binar['T2']==-1)) |
                              ((self.binning.binar['T1']==-1) & (self.binning.binar['T2']==t1i)))[0]
            #If the ndx array is not empty, then append cluster-N counts information to the list
            if len(n_cl)>0: toret.append((t1i, -1, b'+N', None, n_cl))
            
            #Loop over all other proceeding tracers, in case we have 2pt statistics
            for t2i in range(t1i,Nt):
                #Pick out which tracers correspond to tracers t1 and t2
                ndxx=np.where( ((self.binning.binar['T1']==t1i) & (self.binning.binar['T2']==t2i)) |
                              ((self.binning.binar['T1']==t2i) & (self.binning.binar['T2']==t1i)) )[0]
                types=np.unique(self.binning.binar['type'][ndxx])
                #Loop over the types of tracers this 2pt statistic could be, and figure out the ell bins
                for typ in types :
                    ndx=ndxx[self.binning.binar['type'][ndxx]==typ]
                    ells=self.binning.binar['ls'][ndx]
                    if len(ells)>0:
                        toret.append((t1i,t2i,typ,ells,ndx))
        return toret

    def plot_vector (self, tr_number=None, plot_cross=False, set_logx=True, set_logy=True, show_legend=True, out_name=None):
        """
        Plots the mean vector associated to the different tracers
        in the sacc file. The tracers to plot can be selected
        passing a list to tr_number (if None it plots all of them).
        If you want to plot the cross-correlations
        between these bins just set the argument plot_cross to True.
        """      
        import matplotlib.pyplot as plt
        if tr_number is None:
            tr_number=np.arange(len(self.tracers))
        if plot_cross:
            for tr_i in tr_number:
                for tr_j in range(tr_i,len(tr_number)):
                    tbin = np.logical_and(self.binning.binar['T1']==tr_i,self.binning.binar['T2']==tr_j)
                    plt.plot(self.binning.binar['ls'][tbin],self.mean.vector[tbin],'-',label='%i,%i' %(tr_i,tr_j))
        else:
            for tr_i in tr_number:
                tbin = np.logical_and(self.binning.binar['T1']==tr_i,self.binning.binar['T2']==tr_i)
                plt.plot(self.binning.binar['ls'][tbin],self.mean.vector[tbin],'-',label='%i,%i' %(tr_i,tr_i))
        if set_logx:
            plt.xscale('log')
        if set_logy:
            plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_{l}$')
        if show_legend:
            plt.legend(loc='best')
        if out_name is None:
            plt.show()
        else:
            plt.savefig(out_name)
        
    def saveToHDF (self, filename, save_mean=True, save_precision=True, mean_filename=None, precision_filename=None):
        f=h5py.File(filename,'w')
        meta=f.create_dataset("meta",data=[])
        meta.attrs.create("_format_version",self._format_version)
        if self.meta is not None:
            for key,value in (self.meta.items() if PY3 else self.meta.iteritems()):
                if isinstance(value,str) :
                    value=value.encode('ascii')
                meta.attrs.create(key.encode('ascii'),value)
        
        tracer_group=f.create_group("tracers")
        tracer_group.attrs.create("tracer_list",[t.name.encode('ascii') for t in self.tracers])
        for t in self.tracers:
            t.saveToHDF(tracer_group)
        self.binning.saveToHDF(f)
        if save_mean:
            if self.mean is not None:
                self.mean.saveToHDF(f)
        else:
            if mean_filename is not None:
                f.attrs['mean_file_path']=mean_filename

        if save_precision:
            if self.precision is not None:
                self.precision.saveToHDF(f)
        else:
            if precision_filename is not None:
                f.attrs['precision_file_path']=precision_filename
        f.close()
       
    @classmethod
    def loadFromHDF (SACC,filename,mean_filename=None, precision_filename=None):
        f=h5py.File(filename,'r')
        fmeta=f['meta'].attrs
        if (fmeta['_format_version']>SACC._format_version):
            print("Loading file with format version %i on sacc format version %i.",
                  (fmeta['_format_version'],self._format_version))
            raise NotImplementedError()
        meta={}

        for key,value in (fmeta.items() if PY3 else fmeta.iteritems()):
            if key[0]!='_':
                meta[key]=value

        tracer_group=f['tracers']
        tnames=tracer_group.attrs['tracer_list']
        tracers=[Tracer.loadFromHDF(f,n) for n in tnames]
        binning=Binning.loadFromHDF(f)
        ##
        ## if mean specified, we use it, otherwise look for mean group and mean filename
        ## in attributes.
        ##
        mean=None
        if mean_filename is not None:
            fm=h5py.File(mean_filename,'r')
            mean=MeanVec.loadFromHDF(fm)
        else:
            if 'mean' in f.keys():
                mean=MeanVec.loadFromHDF(f)
            else:
                if 'mean_file_path' in f.attrs.keys():
                    fm=h5py.File(f.attrs['mean_file_path'],'r')
                    mean=MeanVec.loadFromHDF(fm)
        precision=None
        if precision_filename is not None:
            fm=h5py.File(precision_filename,'r')
            precision=Precision.loadFromHDF(fm,binning)
        else:
            if 'error' in f.keys():
                precision=Precision.loadFromHDF(f,binning)
            else:
                if 'precision_file_path' in f.attrs.keys():
                    fm=h5py.File(f.attrs['precision_file_path'],'r')
                    precision=Precision.loadFromHDF(fm,binning)
        f.close()
        return SACC(tracers,binning,mean,precision,meta)

                    
        
