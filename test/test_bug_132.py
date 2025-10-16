import os
import tempfile

import numpy as np

import sacc


def test_bug_132():
    s = sacc.Sacc()

    s.add_tracer(
        "Map",
        "tracer",
        quantity="galaxy_density",
        spin=0,
        ell=np.arange(100),
        beam=np.ones(100),
    )

    ndata = 1000
    s.add_ell_cl("cl_00", "tracer", "tracer", np.arange(ndata), np.ones(ndata))
    # Add covariance
    s.add_covariance(np.eye(ndata))

    # Write to SACC files using a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        fits_filename = os.path.join(tmpdir, "test.fits")
        hdf5_filename = os.path.join(tmpdir, "test.hdf5")
        s.save_fits(fits_filename, overwrite=True)
        s.save_hdf5(hdf5_filename, overwrite=True)
        # Read back in
        s2 = sacc.Sacc.load_fits(fits_filename)
        s3 = sacc.Sacc.load_hdf5(hdf5_filename)
        assert s2 == s
        assert s3 == s
