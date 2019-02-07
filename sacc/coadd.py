#
# Helper routine to be imported into module
#

from copy import deepcopy
import numpy as np
import scipy.linalg as la

from .precision import Precision
from .meanvec import MeanVec

def coadd(sacclist, mode='Cinv'):

    if mode == 'Cinv':
        outsacc = coadd_Cinv(sacclist)
    elif mode == 'area':
        outsacc = coadd_area(sacclist)
    else:
        raise NotImplementedError('Only coaddition with inverse-variance and area weighting implemented.')

    return outsacc

def coadd_area(sacclist):
    """ CoAdds sacc object in a list of sacc files and
    returns a new object with area weighted coadded files."""


    if len(sacclist) == 0:
        return None

    outsacc = deepcopy(sacclist[0])
    if len(sacclist) == 1:
        return outsacc

    assert 'Area_rad' in outsacc.meta, 'Need area information in saccfile to perform area weighting.'

    toadd = sacclist[1:]

    w_current = outsacc.meta['Area_rad']
    cw = w_current**2*outsacc.precision.getCovarianceMatrix()
    swd = w_current*outsacc.mean.vector
    w = w_current*1.

    for s in toadd:
        assert(s.mean.vector.shape==outsacc.mean.vector.shape)

        w_current = s.meta['Area_rad']
        c = s.precision.getCovarianceMatrix()

        cw += w_current**2*c
        swd += w_current*s.mean.vector
        w += w_current

        assert (len(outsacc.tracers) == len(s.tracers))
        for otr, ctr in zip(outsacc.tracers, s.tracers):
            for z, zp in zip(otr.z, ctr.z):
                assert(z == zp)
            otr.Nz += ctr.Nz

    newmean = swd/w
    cw /= w**2
    outsacc.precision = Precision(cw, is_covariance=True)
    outsacc.mean = MeanVec(newmean)

    return outsacc

def coadd_Cinv(sacclist):
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

    
