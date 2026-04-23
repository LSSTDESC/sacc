import sacc
import tempfile



def test_lots_of_metadata_fits():
    # Create a new Sacc object
    s = sacc.Sacc()
    # This will fail for N > 1000
    # because of a FITS limitation
    N = 1001
    with tempfile.TemporaryDirectory() as tmpdir:
        # Add some metadata
        for i in range(N):
            s.metadata[f"key_{i}"] = f"value_{i}"

        # Save the Sacc object to a file
        s.save_fits(f"{tmpdir}/test.fits")

        # Load the Sacc object from the file
        s_loaded = sacc.Sacc.load_fits(f"{tmpdir}/test.fits")

        # Check that the metadata is correct
        for i in range(N):
            assert s_loaded.metadata[f"key_{i}"] == f"value_{i}"


def test_lots_of_metadata_hdf():
    # Create a new Sacc object
    s = sacc.Sacc()
    # This will when the header gets too large.
    # For these choices it's at N=1313 but it will 
    # presumably depend on the length of the keys and values.
    N = 1500
    with tempfile.TemporaryDirectory() as tmpdir:
        # Add some metadata
        for i in range(N):
            s.metadata[f"key_{i}"] = f"value_{i}"

        # Save the Sacc object to a file
        s.save_hdf5(f"{tmpdir}/test.hdf")

        # Load the Sacc object from the file
        s_loaded = sacc.Sacc.load_hdf5(f"{tmpdir}/test.hdf")

        # Check that the metadata is correct
        for i in range(N):
            assert s_loaded.metadata[f"key_{i}"] == f"value_{i}"
