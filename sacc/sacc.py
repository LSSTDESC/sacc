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
    """
    SACC creator from its individual components. 
    
    SACC objects are the main container class for 2-point measurements.
    
    :param Tracer tracers: list of `Tracer` objects used in the measurement and referenced in binning parameter
    :param Binning binning: `Binning` object describing what measurement each index in the mean vector and precision matrix contains.
    :param MeanVec mean: Vector representing the actual two-point measurement. If not a MeanVec object, will try to cast it into one.
    :param Precision precision: `Precision` object representing the covariance matrix or its inverse (a.k.a. the precision matrix).
    :param dict meta: dictionary containing additional metadata.
    """
    
    ## format version, bump up every time you break thins
    ## irreparrably
    _format_version=1

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
        """
        Return the set of exp_samples (essentially a label attached to each Tracer) contained in this SACC file.

        :return: set with all the exp_sample labels.
        """
        return set([t.exp_sample for t in self.tracers])
        
    def printInfo(self):
        """
        Prints information about the contents of this SACC file
        """
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
        """
        Returns the size of the mean vector

        :return: vector size.
        """
        return self.binning.size()

    def lrange(self,t1i,t2i):
        """
        Returns the scales at which the cross-correlation between t1i and t2i are stored.

        :param int t1i: index of the first tracer.
        :param int t2i: index of the second tracer.
        :return: array of scales.
        """
        ndx=np.where( ((self.binning.binar['T1']==t1i) & (self.binning.binar['T2']==t2i)) |
                      ((self.binning.binar['T1']==t2i) & (self.binning.binar['T2']==t1i)) )
        return self.binning.binar['ls'][ndx]

    def ilrange(self,t1i,t2i):
        """
        Returns the indices of the data vector containing the cross-correlation between tracers t1i and t2i.

        :param int t1i: index of the first tracer.
        :param int t2i: index of the second tracer.
        :return: array of indices.
        """
        ndx=np.where( ((self.binning.binar['T1']==t1i) & (self.binning.binar['T2']==t2i)) |
                      ((self.binning.binar['T1']==t2i) & (self.binning.binar['T2']==t1i)) )
        return ndx

    def cullLminLmax(self,lmin,lmax):
        """
        Implement scale cuts on the data vector and the covariance.

        :param array_like lmin,lmax: lists of minimum/maximum scales for each tracer. Any correlation will be cut to the most stringent scale cuts associated with the two tracers that go into it. CHECK THAT THIS IS TRUE.
        """
        lmina=np.maximum(
            np.array(lmin)[self.binning.binar['T1']],
            np.array(lmin)[self.binning.binar['T2']])
        lmaxa=np.minimum(
            np.array(lmax)[self.binning.binar['T1']],
            np.array(lmax)[self.binning.binar['T2']])
        ndx=np.where((self.binning.binar['ls']>lmina) &
             (self.binning.binar['ls']<lmaxa))[0]

        self.binning.cullBinning(ndx)
        self.mean.cullVector(ndx)
        self.precision.cullMatrix(ndx)
        
    def cullCross(self):
        """ 
        Cuts out all elements of the data vector (and covariance, etc.) that are not auto-correlations.
        """
        ndx=np.where(self.binning.binar['T1']==self.binning.binar['T2'])[0]
        self.binning.cullBinning(ndx)
        self.mean.cullVector(ndx)
        self.precision.cullMatrix(ndx)
                    
    
    def sortTracers(self):
        """
        Return information about all the ingredients that define each element of the data vector.

        :return: list of tuples, with one element per data vector element. Each tuple contains 5 elements: `t1i`, `t2i`, `typ`, `ells`, `ndx`. `t1i` and `t2i` are the indices of the tracers that go into this cross-correlation. `typ` is the type of correlation contained here. `ells` contains the scales at which this cross-correlation is stored. `ndx` contains the indices of the data vector containing this cross-correlation.
        """
        #TODO: figure out how to make this compatible with 3-pt correlation functions.
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


    def plot_vector (self, subplot = None, plot_corr='all', weightpow = 2, set_logx=True, set_logy=True,
                    show_axislabels = False, show_legend=True, prediction=None, clr='r', lofsf=1.0, label=None):
        """
        Plots the mean vector associated to the different tracers in the SACC file. The tracer correlations to plot can be selected by passing a list of pairs of values in plot_corr.  It can also plot the autocorrelation by passing 'auto', the cross-correlation by passng 'cross', and both by passing 'all'.  The correlations will be weighted by a factor of ell^{weightpow} (where ell is the nominal scale for each cross-correlation element).

        TODO: finish describing all other parameters.
        
        :param str/array_like plot_corr: select which correlations to plot. This can be done by passing a list of tracer index pairs, 'auto' to plot all auto-correlations, 'cross'  to plot all cross-correlations or 'all' to plot everything.
        :param float weightpow: the correlations will be weighted by a factor of ell^{weightpow} (where ell is the nominal scale for each cross-correlation element).
        """      
        import matplotlib.pyplot as plt
        

        if subplot is None:
            fig = plt.figure()
            subplot = fig.add_subplot(111)

        if self.precision is not None:
            errs=np.sqrt(self.precision.getCovarianceMatrix().diagonal())
        else:
            errs=None
        
        plot_cross = False
        plot_auto = False
        plot_pairs = []

        if plot_corr == 'all':
            # Plot the auto-correlation and the cross-correlation
            plot_cross = True
            plot_auto = True
        elif plot_corr == 'cross':
            # Plot ALL cross-correlations only
            plot_cross = True
        elif plot_corr == 'auto':
            # Plot the auto-correlation only
            plot_auto = True
        elif hasattr(plot_corr, '__iter__'):
            plot_pairs = plot_corr
        else:
            print('plot_corr needs to be \'all\', \'auto\',\'cross\', or a list of pairs of values.')

        tracer_array = np.arange(len(self.tracers))
        if plot_cross:
            for tr_i in tracer_array:
                other_tr = np.delete(tracer_array, np.where(tracer_array != tr_i))
                for tr_j in other_tr:
                    # Generate the appropriate list of tracer combinations to plot
                    plot_pairs.append([tr_i, tr_j])

        if plot_auto:
            for tr_i in tracer_array:
                plot_pairs.append([tr_i, tr_i])

        plot_pairs = np.array(plot_pairs)

        ###################################
        # Plotting routines below this line
        ###################################

        for (tr_i, tr_j) in plot_pairs:
            tbin = np.logical_and(self.binning.binar['T1']==tr_i,self.binning.binar['T2']==tr_j)
            ell = self.binning.binar['ls'][tbin]
            C_ell = self.mean.vector[tbin]
            subplot.plot(ell,C_ell * np.power(ell,weightpow),color=clr)
            if errs is not None:
                subplot.errorbar(ell,C_ell * np.power(ell,weightpow),yerr=errs[tbin]*np.power(ell,weightpow), linestyle = 'None',color=clr)
            subplot.scatter(ell,C_ell * np.power(ell,weightpow), s = 20, edgecolor = 'k', c = clr, 
                label= self.tracers[0].exp_sample+' $C_{%i%i}$' %(tr_i,tr_j))
            if prediction is not None:
                subplot.plot(ell,prediction[tbin] * np.power(ell,weightpow),':',color=clr)

        if set_logx:
            subplot.set_xscale('log')
        if set_logy:
            subplot.set_yscale('log')
        if show_axislabels:
            subplot.set_xlabel(r'$l$')
            if weightpow == 0:
                elltext = ''
            elif weightpow == 1:
                elltext = r'$\ell$'
            else:
                elltext = r'$\ell^' + '{%i}$' % weightpow
            subplot.set_ylabel(elltext + r'$C_{l}$')
        if show_legend:
            subplot.legend(loc='best')
        
    def saveToHDF (self, filename, save_mean=True, save_precision=True, mean_filename=None, precision_filename=None):
        """
        Write this SACC object into an HDF5 file.

        :param str filename: path to output file.
        :param boolean save_mean: whether to save the mean vector.
        :param boolean save_precision: whether to save the covariance matrix.
        :param str mean_filename: path to additional output file where the mean is to be saved.
        :param str precision_file: same as `mean_filename` for the precision/covariance matrix.
        """
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
        """
        Create a SACC object from the contents of an HDF5 file.

        :param str filename: path to input file. The file should contain at least tracers and binning.
        :param str mean_filename: path to file containing the data vector (set to None if `filename` already contains this or if you don't nead the data vector).
        :param str precision_filename: same as `mean_filename` for the precision/covariance matrix.
        :return: :class:`SACC` object.
        """
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
