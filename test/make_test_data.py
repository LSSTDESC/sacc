import sacc
import numpy as np
from packaging import version


# Dummy metadata
vecsize = 100
nvec = 2
magic = True
pi = np.pi
greeting = 'Hello, world!'


x = np.linspace(0, 100, vecsize)
y = 1./(x+50)**2

# Create file
s = sacc.Sacc()

# Add tracers
s.add_tracer('NZ', 'nz', x, y)
s.add_tracer('NZ', 'nz_b', x, y)

wth = sacc.TopHatWindow(0., 1.)
ww = sacc.Window(x, y)
s.add_ell_cl('galaxy_density_cl', 'nz', 'nz', x, y, window=[wth] * vecsize)
s.add_theta_xi('galaxy_density_xi', 'nz', 'nz', x, y, window=[ww] * vecsize)


wlth = sacc.LogTopHatWindow(-2., 0.)
s.add_ell_cl('galaxy_density_cl', 'nz_b', 'nz_b', x, y,
                window=[wlth] * vecsize)
nvec += 1

wbpw = sacc.BandpowerWindow(x, np.ones([vecsize, vecsize]))
s.add_tracer('Misc', 'msc')
s.add_tracer('Map', 'map', 0, x, y)
s.add_tracer('NuMap', 'nap', 2, x, y, x, y)

s.add_ell_cl('cl_00', 'nz', 'map', x, y, window=wbpw)
s.add_theta_xi('xi_plus_re', 'msc', 'nap', x, y, window=[wth] * vecsize)
s.add_ell_cl('cl_00', 'map', 'nz_b', x, y, window=wbpw)
nvec += 3

s.metadata['nvec'] = nvec
s.metadata['vecsize'] = vecsize
s.metadata['magic'] = magic
s.metadata['pi'] = pi
s.metadata['greeting'] = greeting

s.add_covariance(np.eye(nvec * vecsize))

if __name__ == '__main__':
    fname_save = 'test/legacy_files/dummy_v' + sacc.__version__ + '.fits'
    s.save_fits(fname_save, overwrite=True)
    fname_save = 'test/legacy_files/dummy_v' + sacc.__version__ + '.hdf5'
    s.save_hdf5(fname_save, overwrite=True)
