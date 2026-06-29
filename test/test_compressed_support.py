#!/usr/bin/env python
"""Tests for reading gzip-compressed FITS and HDF5 SACC files."""

import gzip
import os

import numpy as np
import pytest

import sacc


def create_simple_sacc():
    """Build a small but complete Sacc object used as test data.

    This mirrors the helper in ``test_load_fits.py`` so that the
    compressed-file tests follow the existing pattern.
    """
    s = sacc.Sacc()

    # --- Tracers ---
    s.add_tracer("survey", "my_mock_survey", 100.0)
    s.add_tracer("bin_z", "zbin_0", 0.2, 0.4)
    s.add_tracer("bin_z", "zbin_1", 0.4, 0.6)
    s.add_tracer("bin_richness", "rich_0", 10, 20)
    s.add_tracer("bin_richness", "rich_1", 20, 40)

    # --- Data points ---
    cluster_count = sacc.standard_types.cluster_counts
    rng = np.random.default_rng(42)
    for zbin in ["zbin_0", "zbin_1"]:
        for rbin in ["rich_0", "rich_1"]:
            count_value = rng.integers(50, 150)
            s.add_data_point(
                cluster_count, ("my_mock_survey", zbin, rbin), count_value
            )

    # --- Covariance ---
    ndata = len(s.data)
    covariance = np.diag(np.ones(ndata) * 10)
    s.add_covariance(covariance)

    s.to_canonical_order()
    return s


def _gzip_file(src, dst):
    """Gzip-compress the bytes of ``src`` into the new file ``dst``."""
    with open(src, "rb") as f_in:
        data = f_in.read()
    with gzip.open(dst, "wb") as f_out:
        f_out.write(data)
    return dst


@pytest.fixture
def simple_sacc():
    return create_simple_sacc()


@pytest.fixture
def fits_gz(tmp_path, simple_sacc):
    """A gzip-compressed FITS SACC file named ``*.fits.gz``."""
    plain = os.path.join(tmp_path, "simple.fits")
    simple_sacc.save_fits(plain, overwrite=True)
    return _gzip_file(plain, os.path.join(tmp_path, "simple.fits.gz"))


@pytest.fixture
def hdf5_gz(tmp_path, simple_sacc):
    """A gzip-compressed HDF5 SACC file named ``*.hdf5.gz``."""
    plain = os.path.join(tmp_path, "simple.hdf5")
    simple_sacc.save_hdf5(plain, overwrite=True)
    return _gzip_file(plain, os.path.join(tmp_path, "simple.hdf5.gz"))


@pytest.fixture
def fits_generic_gz(tmp_path, simple_sacc):
    """A gzip-compressed FITS SACC file named generically ``*.sacc.gz``.

    The inner type cannot be inferred from the extension, so detection
    must decompress and inspect the FITS ``SIMPLE`` magic bytes.
    """
    plain = os.path.join(tmp_path, "simple_fits.sacc")
    simple_sacc.save_fits(plain, overwrite=True)
    return _gzip_file(plain, os.path.join(tmp_path, "simple_fits.sacc.gz"))


@pytest.fixture
def hdf5_generic_gz(tmp_path, simple_sacc):
    """A gzip-compressed HDF5 SACC file named generically ``*.sacc.gz``.

    The inner type cannot be inferred from the extension, so detection
    must decompress and inspect the HDF5 signature bytes.
    """
    plain = os.path.join(tmp_path, "simple_hdf5.sacc")
    simple_sacc.save_hdf5(plain, overwrite=True)
    return _gzip_file(plain, os.path.join(tmp_path, "simple_hdf5.sacc.gz"))


# ---------------------------------------------------------------------------
# detect_sacc_file_type
# ---------------------------------------------------------------------------

def test_detect_compressed_fits_by_extension(fits_gz):
    assert sacc.utils.detect_sacc_file_type(fits_gz) == "fits"


def test_detect_compressed_hdf5_by_extension(hdf5_gz):
    assert sacc.utils.detect_sacc_file_type(hdf5_gz) == "hdf5"


def test_detect_compressed_fits_by_content(fits_generic_gz):
    # ``.sacc.gz`` gives no inner-type hint, so detection must sniff the
    # decompressed FITS magic bytes.
    assert sacc.utils.detect_sacc_file_type(fits_generic_gz) == "fits"


def test_detect_compressed_hdf5_by_content(hdf5_generic_gz):
    # ``.sacc.gz`` gives no inner-type hint, so detection must sniff the
    # decompressed HDF5 signature bytes.
    assert sacc.utils.detect_sacc_file_type(hdf5_generic_gz) == "hdf5"


# ---------------------------------------------------------------------------
# Reading compressed files
# ---------------------------------------------------------------------------

def test_load_fits_reads_compressed(fits_gz, simple_sacc):
    loaded = sacc.Sacc.load_fits(fits_gz)
    assert loaded == simple_sacc


def test_load_hdf5_reads_compressed(hdf5_gz, simple_sacc):
    loaded = sacc.Sacc.load_hdf5(hdf5_gz)
    assert loaded == simple_sacc


def test_generic_load_reads_compressed_fits(fits_gz, simple_sacc):
    loaded = sacc.Sacc.load(fits_gz)
    assert loaded == simple_sacc


def test_generic_load_reads_compressed_hdf5(hdf5_gz, simple_sacc):
    loaded = sacc.Sacc.load(hdf5_gz)
    assert loaded == simple_sacc


def test_generic_load_reads_compressed_fits_by_content(fits_generic_gz, simple_sacc):
    loaded = sacc.Sacc.load(fits_generic_gz)
    assert loaded == simple_sacc


def test_generic_load_reads_compressed_hdf5_by_content(hdf5_generic_gz, simple_sacc):
    loaded = sacc.Sacc.load(hdf5_generic_gz)
    assert loaded == simple_sacc


def test_compressed_and_uncompressed_agree(tmp_path, simple_sacc):
    # The same source object, saved uncompressed and compressed, must load
    # to equal Sacc objects in both FITS and HDF5 formats.
    plain_fits = os.path.join(tmp_path, "agree.fits")
    plain_hdf5 = os.path.join(tmp_path, "agree.hdf5")
    simple_sacc.save_fits(plain_fits, overwrite=True)
    simple_sacc.save_hdf5(plain_hdf5, overwrite=True)

    gz_fits = _gzip_file(plain_fits, os.path.join(tmp_path, "agree.fits.gz"))
    gz_hdf5 = _gzip_file(plain_hdf5, os.path.join(tmp_path, "agree.hdf5.gz"))

    assert sacc.Sacc.load_fits(plain_fits) == sacc.Sacc.load_fits(gz_fits)
    assert sacc.Sacc.load_hdf5(plain_hdf5) == sacc.Sacc.load_hdf5(gz_hdf5)


# ---------------------------------------------------------------------------
# Cross-type errors on compressed files
# ---------------------------------------------------------------------------

def test_load_fits_rejects_compressed_hdf5(hdf5_gz):
    with pytest.raises(
        ValueError,
        match=r"is of type hdf5, not fits\. Use Sacc\.load or Sacc\.load_hdf5 to load it",
    ):
        sacc.Sacc.load_fits(hdf5_gz)


def test_load_hdf5_rejects_compressed_fits(fits_gz):
    with pytest.raises(
        ValueError,
        match=r"is of type fits, not hdf5\. Use Sacc\.load or Sacc\.load_fits to load it",
    ):
        sacc.Sacc.load_hdf5(fits_gz)
