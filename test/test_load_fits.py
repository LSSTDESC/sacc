#!/usr/bin/env python

import numpy as np
import sacc
import pytest


def create_simple_sacc():
    # Create a new Sacc object
    s = sacc.Sacc()

    # --- 1️⃣ Define tracers ---
    # Add a survey tracer (usually represents the data source)
    s.add_tracer("survey", "my_mock_survey", 100.0)  # 100 deg^2 area, for example

    # Add two bin tracers (e.g., redshift and richness bins)
    s.add_tracer("bin_z", "zbin_0", 0.2, 0.4)
    s.add_tracer("bin_z", "zbin_1", 0.4, 0.6)
    s.add_tracer("bin_richness", "rich_0", 10, 20)
    s.add_tracer("bin_richness", "rich_1", 20, 40)
    print(type(s.tracers["zbin_0"]))

    # --- 2️⃣ Add mock data ---
    # We’ll use a standard type — cluster counts
    cluster_count = sacc.standard_types.cluster_counts

    # reproducible random numbers
    rng = np.random.default_rng(42)

    for zbin in ["zbin_0", "zbin_1"]:
        for rbin in ["rich_0", "rich_1"]:
            count_value = rng.integers(50, 150)  # random counts
            s.add_data_point(cluster_count, ("my_mock_survey", zbin, rbin), count_value)

    # --- 3️⃣ Add covariance ---
    # Create a simple diagonal covariance matrix (variance = 10 for each data point)
    ndata = len(s.data)
    covariance = np.diag(np.ones(ndata) * 10)
    s.add_covariance(covariance)

    # --- 4️⃣ Save the SACC file ---
    s.to_canonical_order()
    return s


def test_bool_numpy_error():
    s = create_simple_sacc()
    s.save_fits("test/data/simple_mock_clusters.sacc", overwrite=True)
    print(f"SACC file saved with {len(s.data)} data points: test/data/simple_mock_clusters.sacc")
    t2 = sacc.Sacc.load_fits("test/data/simple_mock_clusters.sacc")


def test_load_any():
    s = create_simple_sacc()
    fits_filename = "test/data/simple_mock_clusters.fits"
    hdf5_filename = "test/data/simple_mock_clusters.hdf5"
    generic_filename = "test/data/simple_mock_clusters.sacc"

    # Check the generic loader works when the file extension is explicitly .fits
    s.save_fits(fits_filename, overwrite=True)
    assert sacc.utils.detect_sacc_file_type(fits_filename) == "fits"
    sacc.Sacc.load_fits(fits_filename)

    # and the same check for HDF5
    s.save_hdf5(hdf5_filename, overwrite=True)
    assert sacc.utils.detect_sacc_file_type(hdf5_filename) == "hdf5"
    sacc.Sacc.load_hdf5(hdf5_filename)

    s.save_fits(generic_filename, overwrite=True)
    assert sacc.utils.detect_sacc_file_type(generic_filename) == "fits"
    with pytest.raises(ValueError, match="is of type fits, not hdf5\. Use Sacc\.load or Sacc\.load_fits to load it"):
        sacc.Sacc.load_hdf5(generic_filename)

    # Load with the generic method, which should detect the format
    t1 = sacc.Sacc.load(generic_filename)
    t1a = sacc.Sacc.load_fits(generic_filename)

    s.save_hdf5(generic_filename, overwrite=True)
    assert sacc.utils.detect_sacc_file_type(generic_filename) == "hdf5"
    with pytest.raises(ValueError, match="is of type hdf5, not fits\. Use Sacc\.load or Sacc\.load_hdf5 to load it"):
        sacc.Sacc.load_fits(generic_filename)

    # should be able to load from either format with Sacc.load
    t2 = sacc.Sacc.load(generic_filename)
    t2a = sacc.Sacc.load_hdf5(generic_filename)

    # All the loaded objects should be identical
    assert t1 == t2
    assert t1a == t2a
    assert t1 == t1a
    # and be equal to the original data
    assert t1 == s
