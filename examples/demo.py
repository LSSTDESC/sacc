
import numpy as np
from sacc import Sacc, BandpowerWindow

# Create Sacc object
s = Sacc()

# Add metadata
s.metadata['creator'] = 'Demo'
s.metadata['purpose'] = 'Full feature test'

 # Add tracers
z = np.linspace(0, 1, 101)
nz = np.exp(-((z-0.5)/0.1)**2)
s.add_tracer('NZ', 'source_0', z, nz)
s.add_tracer('NZ', 'source_1', z, nz)

nu = np.linspace(30, 60, 5)
bandpass = np.ones(5)

ell = np.linspace(10, 1000, 100)
beam = np.exp(-0.01 * ell)
s.add_tracer('NuMap', 'cmb', 2, nu, bandpass, ell, beam)









# Add window (correct shape)
ells_large = np.arange(100)
window_matrix = np.eye(100)  # shape (100, 100)
win1 = BandpowerWindow(ells_large, window_matrix)

# Add data points, referencing the window object directly
for i in range(100):
    s.add_data_point('cl_ee', ('source_0', 'source_1'), 0.1*i, ell=ell[i], window=win1)
    s.add_data_point('cl_00', ('source_0', 'cmb'), 0.2*i, ell=ell[i], window=win1)

# Add covariance
n = len(s.data)
cov = np.eye(n)
s.add_covariance(cov)

# Write to files
s.save_fits('demo.fits', overwrite=True)
s.save_hdf5('demo.hdf5', overwrite=True)
