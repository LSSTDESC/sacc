#!/usr/bin/env python

import sacc
import numpy as np

## let's generate four redshift bins for two galaxy populations each.
tracers=[]
for i,z in enumerate([0.3,0.5,0.7,0.9]):
    zar=np.arange(z-0.1,z+0.1,0.001)
    Nz=np.exp(-(z-zar)**2/(2*0.03**2))
    bias=np.ones(len(zar))*(i+0.5)
    T=sacc.Tracer("des_gals_"+str(i),"point",zar,Nz,exp_sample="des_gals",
                               Nz_sigma_logmean=0.01, Nz_sigma_logwidth=0.1)
    T.addColumns({'b':bias})
    tracers.append(T)
    
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
    T=sacc.Tracer("lsst_gals_"+str(i),"point",zar,Nz,exp_sample="lsst_gals",
                               DNz=DNz)
    T.addColumns({'b':bias})
    tracers.append(T)

# and also add CMB
tracers.append (sacc.Tracer("Planck","cmb", None, None))

# Now, let's have cross-correlation of everything with everything
# dummy values
lvals=np.arange(100,1000,100)
Ntracer=len(tracers)
type,ell,t1,q1,t2,q2,val,err=[],[],[],[],[],[],[],[]
for t1i in range(Ntracer):
    for t2i in range(t1i,Ntracer):
        for l in lvals:
            type.append('F')
            ell.append(l)
            t1.append(t1i)
            q1.append('P' if t1i<8 else 'I') ##last is CMB
            t2.append(t2i)
            q2.append('P' if t2i<8 else 'I')
            val.append(np.random.uniform(0,10))
            err.append(np.random.uniform(0,1))

binning=sacc.Binning(type,ell,t1,q1,t2,q2)
mean=sacc.MeanVec(val)

## now covariance matrix
Np=binning.size()
icov=np.zeros ((Np,Np))
for i in range(Np):
    for j in range (i,Np):
        if ell[i]==ell[j]:
            icov[i,j]=1/(err[i]*err[j])
            if (i!=j):
                icov[i,j]/=10
            icov[j,i]=icov[i,j]
precision=sacc.Precision(icov,"ell_block_diagonal",binning)
            

# create SACC object
s=sacc.SACC(tracers,binning,mean,precision)
s.printInfo()
s.saveToHDF ("test.sacc")
