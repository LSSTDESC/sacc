#!/usr/bin/env python

import numpy as np
import sacc
from astropy.io import fits

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

    rng = np.random.default_rng(42)  # reproducible random numbers

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
    s.save_fits("simple_mock_clusters.sacc", overwrite=True)

    print(f"SACC file saved with {ndata} data points: simple_mock_clusters.sacc")

if __name__ == "__main__":
    create_simple_sacc()
fits.info("simple_mock_clusters.sacc")
t2 = sacc.Sacc.load_fits("simple_mock_clusters.sacc")
print("\n\n")
print(sacc.__version__)