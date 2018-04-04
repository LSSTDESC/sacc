#!/usr/bin/env python

import sacc
import numpy as np

## We will write the following toy example:.
##
## We have some DES galaxies and we also have some LSST galaxies and the CMB kappa map
##
## we start by defining tracers
##

tracers=[]

## First DES galaxies with 4 tomographic bins:
for i,z in enumerate([0.3,0.5,0.7,0.9]):
    zar=np.arange(z-0.1,z+0.1,0.001)
    Nz=np.exp(-(z-zar)**2/(2*0.03**2))
    bias=np.ones(len(zar))*(i+0.5)
    T=sacc.Tracer(b"des_gals_%i"%i,b"point",zar,Nz,exp_sample=b"des_gals",
                               Nz_sigma_logmean=0.01, Nz_sigma_logwidth=0.1)
    T.addColumns({'b':bias})
    tracers.append(T)

## Next LSS galaxies with 4 different tomographic bins.
## Here the PZ modelling got more advanced so we have some PZ shapes to marginalise over

for i,z in enumerate([0.5,0.7,0.9,1.1]):
    zar=np.arange(z-0.1,z+0.1,0.001)
    Nz=np.exp(-(z-zar)**2/(2*0.025**2))
    DNz=np.zeros((len(Nz),2))
    ## some random shapes of Nz to marginalise over
    DNz[:,0]=(z-zar)**2*0.01
    DNz[:,0]-=DNz[:,0].mean()
    DNz[:,1]=(z-zar)**3*0.01
    DNz[:,1]-=DNz[:,1].mean()
    bias=np.ones(len(zar))*(i+0.7)
    T=sacc.Tracer(b"lsst_gals_%i"%i,b"point",zar,Nz,exp_sample=b"lsst_gals",
                               DNz=DNz)
    T.addColumns({'b':bias})
    tracers.append(T)

# and also add CMB
tracers.append (sacc.Tracer(b"Planck","cmb", None, None))

# Now, let's have cross-correlation of everything with everything
# at 100 ell bins for density correlations
lvals=np.arange(100,1000,100)
Ntracer=len(tracers)
type,ell,t1,q1,t2,q2,val,err,wins=[],[],[],[],[],[],[],[],[]
for t1i in range(Ntracer):
    for t2i in range(t1i,Ntracer):
        for l in lvals:
            ## we have Fourier space measurement
            type.append('F')
            ## at this nominal ell
            ell.append(l)
            ## but in detail  the measurement
            ## is a Gaussian window around central ell +/- 50
            wins.append(sacc.Window(np.arange(l-50,l+50),np.exp(-(1.0*np.arange(-50,50))**2/(2*20.**2))))
            ## We refer to tracers by their index
            t1.append(t1i)
            t2.append(t2i)
            ## Here we have density cross-correlations so "P" as point
            ## except for CMB where 
            q1.append('P' if t1i<8 else 'K') ##last is CMB, where we have kappa
            q2.append('P' if t2i<8 else 'K')
            ## values and errors
            val.append(np.random.uniform(0,10))
            err.append(np.random.uniform(1,2))

binning=sacc.Binning(type,ell,t1,q1,t2,q2,windows=wins)
mean=sacc.MeanVec(val)


## We need to add covariance matrix. We will use ell_block_diagonal
## where everything is coupled across tracers/redshifts at the same ell but not
## across ell with fixed 10% off-diagonal elements
Np=binning.size()
cov=np.zeros ((Np,Np))
for i in range(Np):
    for j in range (i,Np):
        if ell[i]==ell[j]:
            cov[i,j]=err[i]*err[j]
            if (i!=j):
                cov[i,j]/=10
            cov[j,i]=cov[i,j]
precision=sacc.Precision(cov,"ell_block_diagonal",is_covariance=True, binning=binning)
            
## Add some meta data
meta={b'Creator':b'McGyver',b'Project':b'Victory'}


## finally, create SACC object
s=sacc.SACC(tracers,binning,mean,precision,meta)
s.printInfo()
s.saveToHDF ("test.sacc")
