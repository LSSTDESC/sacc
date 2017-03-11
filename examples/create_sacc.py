#!/usr/bin/env python

import sacc
import numpy as np

## let's generate four redshift bins for two galaxy populations each.
tracers=[]
for i,z in enumerate([0.3,0.5,0.7,0.9]):
    zar=np.arange(z-0.1,z+0.1,0.001)
    Nz=np.exp(-(z-zar)**2/(2*0.03**2))
    tracers.append(sacc.Tracer("des_gals_"+str(i),"point",zar,Nz,exp_sample="des_gals",
                               Nz_sigma_logmean=0.01, Nz_sigma_logwidth=0.1))
for i,z in enumerate([0.5,0.7,0.9,1.1]):
    zar=np.arange(z-0.1,z+0.1,0.001)
    Nz=np.exp(-(z-zar)**2/(2*0.025**2))
    DNz=np.zeros((len(Nz),2))
    ## some random shapes of Nz to marginalise over
    DNz[:,0]=(z-zar)**2*0.01
    DNz[:,0]-=DNz[:,0].mean()
    DNz[:,1]=(z-zar)**3*0.01
    DNz[:,1]-=DNz[:,1].mean()
    tracers.append(sacc.Tracer("lsst_gals_"+str(i),"point",zar,Nz,exp_sample="lsst_gals",
                               DNz=DNz))
# and also add CMB
tracers.append (sacc.Tracer("Planck","cmb", None, None))

# create SACC object
s=sacc.SACC(tracers)
s.saveToHDF ("test.sacc")


                

    
