#
# Helper routine to be imported into module
#

from copy import deepcopy
import numpy as np
import scipy.linalg as la

from .precision import Precision
from .meanvec import MeanVec
def coadd(sacclist):
    """ CoAdds sacc object in a list of sacc files and
    returns a new object with Cinverse coadded files."""


    if len(sacclist)==0:
        return None

    outsacc=deepcopy(sacclist[0])
    if len(sacclist)==1:
        return outsacc

    toadd=sacclist[1:]
    sw=outsacc.precision.getPrecisionMatrix()
    swd=np.dot(sw,outsacc.mean.vector)
    for s in toadd:
        assert(s.mean.vector.shape==outsacc.mean.vector.shape)
        p=s.precision.getPrecisionMatrix()
        sw+=p
        swd+=np.dot(p,s.mean.vector)
        assert (len(outsacc.tracers)==len(s.tracers))
        for otr,ctr in zip(outsacc.tracers,s.tracers):
            for z,zp in zip(otr.z,ctr.z):
                assert(z==zp)
            otr.Nz+=ctr.Nz
        
    newmean=np.dot(la.inv(sw),swd)
    outsacc.precision=Precision(sw, is_covariance=False)
    outsacc.mean=MeanVec(newmean)
    return outsacc

    
