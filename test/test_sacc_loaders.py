"""
Tests targeting uncovered branches in Sacc.load_fits and Sacc.load_hdf5.
"""

import warnings

import h5py
import numpy as np
import pytest
from astropy.io import fits

from sacc.sacc import Sacc, SACCFVER, SACCHDF5VER


# ---------------------------------------------------------------------------
# load_fits – line 996
# RuntimeError when SACCFVER in the file exceeds the library's SACCFVER.
# A primary-only FITS file is sufficient; the version check is the very first
# thing done after opening the file.
# ---------------------------------------------------------------------------

class TestLoadFitsFutureVersion:

    def test_raises_runtime_error_for_future_fits_version(self, tmp_path):
        """Line 996: fitsver > SACCFVER triggers RuntimeError."""
        future_version = SACCFVER + 1

        primary = fits.PrimaryHDU()
        primary.header['SACCFVER'] = future_version
        hdul = fits.HDUList([primary])

        path = str(tmp_path / 'future_version.fits')
        hdul.writeto(path)

        with pytest.raises(
            RuntimeError,
            match=f"Unsupported SACC FITS version: {future_version}",
        ):
            Sacc.load_fits(path)


# ---------------------------------------------------------------------------
# load_fits – line 1002
# Legacy NMETA header: key/value metadata pairs embedded in the primary HDU.
# A primary-only FITS file is sufficient; the metadata is extracted from the
# primary header before any data HDUs are processed, and ``from_tables([])``
# happily returns an empty Sacc.
# ---------------------------------------------------------------------------

class TestLoadFitsLegacyNmeta:

    def test_metadata_loaded_from_nmeta_header(self, tmp_path):
        """Line 1002: NMETA > 0 causes KEY*/VAL* pairs to be read into metadata."""
        primary = fits.PrimaryHDU()
        # Deliberately omit SACCFVER so fitsver falls back to 1 (≤ SACCFVER == 2)
        primary.header['NMETA'] = 2
        primary.header['KEY0'] = 'author'
        primary.header['VAL0'] = 'tester'
        primary.header['KEY1'] = 'project'
        primary.header['VAL1'] = 'sacc'
        hdul = fits.HDUList([primary])

        path = str(tmp_path / 'nmeta.fits')
        hdul.writeto(path)

        s = Sacc.load_fits(path)

        assert s.metadata.get('author') == 'tester'
        assert s.metadata.get('project') == 'sacc'

    def test_nmeta_zero_produces_no_metadata(self, tmp_path):
        """NMETA = 0: loop body is skipped, metadata dict stays empty."""
        primary = fits.PrimaryHDU()
        primary.header['NMETA'] = 0
        hdul = fits.HDUList([primary])

        path = str(tmp_path / 'nmeta_zero.fits')
        hdul.writeto(path)

        s = Sacc.load_fits(path)
        assert s.metadata == {}


# ---------------------------------------------------------------------------
# load_fits – line 1004
# Legacy covariance HDU: an HDU named exactly 'covariance' is handled by
# BaseCovariance.from_hdu (the old image-based path) rather than the modern
# table-based path.  The modern writer names it 'covariance:full:cov', which
# does NOT match this branch, so no existing saved file exercises it.
#
# Fixture: primary HDU (SACCFVER=2) + ImageHDU named 'covariance' holding a
# 0×0 float array.  Zero data points means a 0×0 covariance is valid, and no
# BinTableHDU with string columns is needed — entirely avoiding the
# numpy/astropy environment bug.
# ---------------------------------------------------------------------------

class TestLoadFitsLegacyCovarianceHdu:

    def test_legacy_covariance_hdu_is_loaded(self, tmp_path):
        """Line 1004: HDU named 'covariance' is processed by BaseCovariance.from_hdu."""
        primary = fits.PrimaryHDU()
        primary.header['SACCFVER'] = 2

        # 0×0 identity matches zero data points; no BinTableHDUs required
        C = np.zeros((0, 0), dtype=float)
        cov_hdu = fits.ImageHDU(data=C, name='covariance')
        cov_hdu.header['saccclss'] = 'full'

        hdul = fits.HDUList([primary, cov_hdu])
        path = str(tmp_path / 'legacy_cov.fits')
        hdul.writeto(path)

        with pytest.warns(DeprecationWarning, match="older SACC legacy"):
            s = Sacc.load_fits(path)

        assert s.has_covariance()
        assert s.covariance.dense.shape == (0, 0)


# ---------------------------------------------------------------------------
# load_hdf5 – line 1122
# When 'sacc_hdf5_version' is absent the version silently defaults to 1.
# An empty HDF5 file (no datasets at all) satisfies this condition; from_tables
# with no tables returns an empty-but-valid Sacc.
# ---------------------------------------------------------------------------

class TestLoadHdf5MissingVersionKey:

    def test_loads_successfully_when_version_key_absent(self, tmp_path):
        """Line 1122: absent 'sacc_hdf5_version' causes hdf5ver to default to 1."""
        path = str(tmp_path / 'no_version.hdf5')

        with h5py.File(path, 'w') as f:
            pass  # intentionally write nothing

        s = Sacc.load_hdf5(path)

        assert isinstance(s, Sacc)
        assert len(s) == 0
        assert s.tracers == {}
        assert not s.has_covariance()


# ---------------------------------------------------------------------------
# load_hdf5 – line 1124
# RuntimeError when sacc_hdf5_version in the file is greater than SACCHDF5VER.
# ---------------------------------------------------------------------------

class TestLoadHdf5FutureVersion:

    def test_raises_runtime_error_for_future_hdf5_version(self, tmp_path):
        """Line 1124: hdf5ver > SACCHDF5VER triggers RuntimeError."""
        future_version = SACCHDF5VER + 1
        path = str(tmp_path / 'future_version.hdf5')

        with h5py.File(path, 'w') as f:
            f.create_dataset(
                'sacc_hdf5_version',
                data=np.array([future_version], dtype='i4'),
            )

        with pytest.raises(
            RuntimeError,
            match=f"Unsupported SACC HDF5 version: {future_version}",
        ):
            Sacc.load_hdf5(path)
