#!/usr/bin/env python
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sacc

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
    T=sacc.Tracer("des_gals_%i"%i,"spin0",zar,Nz,exp_sample="des_gals",
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
    T=sacc.Tracer("lsst_gals_%i"%i,"spin0",zar,Nz,exp_sample="lsst_gals",
                               DNz=DNz)
    T.addColumns({'b':bias})
    tracers.append(T)

# and also add CMB
tracers.append (sacc.Tracer("Planck","spin0", None, None))

## Clusters with 5 richness bins and 4 tomographic bins:
lambda_bins = [20,30,50,80,120,180]
for i,z in enumerate([0.3,0.5,0.7,0.9]):
    zar=np.arange(z-0.1,z+0.1,0.001)
    Nz=np.exp(-(z-zar)**2/(2*0.03**2))
    for j in range(0,len(lambda_bins)-1):
        l_min = lambda_bins[j]
        l_max = lambda_bins[j+1]

        T=sacc.Tracer("clusters_z%i_l%i"%(i,j),"spin0",zar,Nz,exp_sample="clusters", Mproxy_name = "richness", Mproxy_min = l_min, Mproxy_max = l_max)
        tracers.append(T)

## source galaxies with 4 redshift bins
for i,z in enumerate([0.5,0.7,0.9,1.1]):
    zar=np.arange(z-0.1,z+0.1,0.001)
    Nz=np.exp(-(z-zar)**2/(2*0.025**2))
    DNz=np.zeros((len(Nz),2))
    ## some random shapes of Nz to marginalise over
    DNz[:,0]=(z-zar)**2*0.01
    DNz[:,0]-=DNz[:,0].mean()
    DNz[:,1]=(z-zar)**3*0.01
    DNz[:,1]-=DNz[:,1].mean()
    T=sacc.Tracer("lsst_sources_%i"%i,"spin2",zar,Nz,exp_sample="lsst_sources",
                               DNz=DNz)
    tracers.append(T)


# Now, let's have cross-correlation of everything with everything
# at 100 ell bins for density correlations
lvals=np.arange(100,1000,100)
# and at 10 radial bins for cluster weak lensing
rvals=np.logspace(np.log10(0.5), np.log10(3), 10)
Ntracer=len(tracers)
type,bins,t1,q1,t2,q2,val,err,wins=[],[],[],[],[],[],[],[],[]
for t1i in range(Ntracer):
    if tracers[t1i].is_CL():
        for r in rvals:
            ## we have number counts
            type.append('+N')
            bins.append(-1.)
            ## We refer to tracers by their index
            t1.append(t1i)
            t2.append(-1)
            ## Here we have cluster number counts
            q1.append('S')
            q2.append('0')
            ## values and errors
            val.append(t1i+1)
            err.append(np.sqrt(float(t1i+1)))
        for t2i in range(t1i,Ntracer):
            if tracers[t2i].is_WL():
                for r in rvals:
                    ## we have configuration space measurement
                    type.append('+R')
                    ## at this nominal rbins
                    bins.append(r)
                    ## We refer to tracers by their index
                    t1.append(t1i)
                    t2.append(t2i)
                    ## Here we have density-shear cross-correlations
                    q1.append('S')
                    q2.append('E')
                    ## values and errors
                    val.append(np.random.uniform(0,10))
                    err.append(np.random.uniform(1,2))
    else:
        for t2i in range(t1i,Ntracer):
            for l in lvals:
                ## we have Fourier space measurement
                type.append('FF')
                ## at this nominal ell
                bins.append(l)
                ## but in detail  the measurement
                ## is a Gaussian window around central ell +/- 50
                wins.append(sacc.Window(np.arange(l-50,l+50),np.exp(-(1.0*np.arange(-50,50))**2/(2*20.**2))))
                ## We refer to tracers by their index
                t1.append(t1i)
                t2.append(t2i)
                ## Here we have density cross-correlations so "P" as point
                ## except for CMB where 
                q1.append('S')
                q2.append('S')
                ## values and errors
                val.append(np.random.uniform(0,10))
                err.append(np.random.uniform(1,2))

        
binning=sacc.Binning(type,bins,t1,q1,t2,q2,windows=wins)
mean=sacc.MeanVec(val)


## We need to add covariance matrix. We will use ell_block_diagonal
## where everything is coupled across tracers/redshifts at the same ell but not
## across ell with fixed 10% off-diagonal elements
Np=binning.size()
cov=np.zeros ((Np,Np))
for i in range(Np):
    for j in range (i,Np):
        if binning.binar['type'][i] == '+N':
            if i == j:
                cov[i,i] = err[i]*err[i]
        if binning.binar['type'][i] == '+R':
            cov[i,j] = err[i]*err[j]
            if i != j:
                cov[i,j] /= 10.
                cov[j,i] = cov[i,j]
        else:
            if bins[i]==bins[j]:
                cov[i,j]=err[i]*err[j]
                if (i!=j):
                    cov[i,j]/=10
                cov[j,i]=cov[i,j]
precision=sacc.Precision(cov,"ell_block_diagonal",is_covariance=True, binning=binning)
            
## Add some meta data
meta={"Creator":"McGyver","Project":"Victory"}


## finally, create SACC object
s=sacc.SACC(tracers,binning,mean,precision,meta)
s.printInfo()
s.saveToHDF ("test.sacc")
