import sacc
import numpy as np

## We will write the following toy example:.
##
## We have clusters and define both number counts and weak lensing signal
##
## we start by defining tracers
##

tracers=[]
lambda_bins = [20,30,50,80,120,180]

## First clusters with 5 richness bins and 4 tomographic bins:
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

for t in tracers:
    print(t.name)

# define bins for lensing signal
rvals=np.logspace(np.log10(0.5), np.log10(3), 10)
Ntracer=len(tracers)
type,rbins,t1,q1,t2,q2,val,err=[],[],[],[],[],[],[],[]
for t1i in range(Ntracer):
    if tracers[t1i].is_CL():
        for t2i in range(t1i,Ntracer):
            if tracers[t2i].is_WL():
                for r in rvals:
                    ## we have configuration space measurement
                    type.append('+R')
                    ## at this nominal rbins
                    rbins.append(r)
                    ## We refer to tracers by their index
                    t1.append(t1i)
                    t2.append(t2i)
                    ## Here we have density-shear cross-correlations
                    q1.append('S')
                    q2.append('E')
                    ## values and errors
                    val.append(np.random.uniform(0,10))
                    err.append(np.random.uniform(1,2))

# define number counts
for t1i in range(Ntracer):
    if tracers[t1i].is_CL():
        for r in rvals:
            ## we have number counts
            type.append('+N')
            rbins.append(-1.)
            ## We refer to tracers by their index
            t1.append(t1i)
            t2.append(-1)
            ## Here we have cluster number counts
            q1.append('S')
            q2.append('0')
            ## values and errors
            val.append(t1i+1)
            err.append(np.sqrt(float(t1i+1)))
        
        
binning=sacc.Binning(type,rbins,t1,q1,t2,q2,windows=None)
mean=sacc.MeanVec(val)

## Covariance matrix
Np=binning.size()
cov=np.zeros ((Np,Np))
for i in range(Np):
    for j in range(Np):
        if binning.binar['type'][i] == '+N':
            if i == j:
                cov[i,i] = err[i]*err[i]
        if binning.binar['type'][i] == '+R':
            cov[i,j] = err[i]*err[j]
            if i != j:
                cov[i,j] /= 10.
                cov[j,i] = cov[i,j]

precision=sacc.Precision(cov,"dense",is_covariance=True, binning=binning)

## Add some meta data
meta={"Creator":"McGyver","Project":"Victory"}

## finally, create SACC object
s=sacc.SACC(tracers,binning,mean,precision,meta)
s.printInfo()
s.saveToHDF ("test_clusters.sacc")
