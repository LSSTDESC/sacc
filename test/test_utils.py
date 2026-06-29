"""
Tests for ``decompress_gzip_to_tempfile`` and ``detect_sacc_file_type``
in ``sacc.utils``, targeting the branches not yet covered by the existing
test suite.

Missing lines identified from coverage.json (run 2026-06-28):
  decompress_gzip_to_tempfile : 240, 247
  detect_sacc_file_type       : 338, 339
"""

import gzip
import os
import stat

import pytest

from sacc.utils import decompress_gzip_to_tempfile, detect_sacc_file_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gzip(path, content=b"data"):
    with gzip.open(path, "wb") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# decompress_gzip_to_tempfile
# ---------------------------------------------------------------------------

class TestDecompressGzipToTempfile:

    # --- line 240: FileNotFoundError when the source file does not exist ---

    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.fits.gz")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            decompress_gzip_to_tempfile(missing)

    # --- line 247: else branch — filename does NOT end with '.gz' ----------
    # (the function still works; it just keeps the basename as-is for the
    #  temp-file prefix, rather than stripping the .gz suffix)

    def test_non_gz_filename_uses_basename_as_prefix(self, tmp_path):
        # Write a file that gzip can open (i.e. valid gzip content) but whose
        # name does not end in '.gz', so the else branch (line 247) is taken.
        plain = tmp_path / "myfile.bin"
        _write_gzip(plain, b"hello")          # valid gzip, just not named .gz

        temp_path = decompress_gzip_to_tempfile(str(plain))
        try:
            assert os.path.exists(temp_path)
            with open(temp_path, "rb") as fh:
                assert fh.read() == b"hello"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --- sanity: normal .gz path still works (regression guard) -----------

    def test_normal_gz_decompression(self, tmp_path):
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"fits content")

        temp_path = decompress_gzip_to_tempfile(str(gz))
        try:
            assert os.path.exists(temp_path)
            with open(temp_path, "rb") as fh:
                assert fh.read() == b"fits content"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ---------------------------------------------------------------------------
# detect_sacc_file_type
# ---------------------------------------------------------------------------

class TestDetectSaccFileType:

    # --- line 335: content-sniff returns 'hdf5' for HDF5 magic bytes -------

    def test_hdf5_magic_bytes_detected_by_content(self, tmp_path):
        # A file with an unrecognised extension forces the content-sniff path.
        # Writing the HDF5 signature at the start hits line 335.
        generic = tmp_path / "data.sacc"
        generic.write_bytes(b"\x89HDF\r\n\x1a\nextra bytes here")

        assert detect_sacc_file_type(str(generic)) == "hdf5"

    # --- line 337: content-sniff returns 'fits' for FITS magic bytes -------

    def test_fits_magic_bytes_detected_by_content(self, tmp_path):
        # Writing a FITS-style "SIMPLE  " header at the start hits line 337.
        generic = tmp_path / "data.sacc"
        generic.write_bytes(b"SIMPLE  = T")

        assert detect_sacc_file_type(str(generic)) == "fits"

    # --- line 341: content-sniff reads file but bytes match nothing --------

    def test_unknown_content_raises_value_error(self, tmp_path):
        generic = tmp_path / "data.sacc"
        generic.write_bytes(b"not a sacc file at all")

        with pytest.raises(ValueError, match="Could not detect file type"):
            detect_sacc_file_type(str(generic))

    # --- lines 338-339: except branch — file exists but cannot be read ----
    # We trigger this by making the file unreadable after creation so that
    # open() raises a PermissionError, which is caught and re-raised as
    # ValueError.

    @pytest.mark.skipif(
        os.getuid() == 0,
        reason="root bypasses file-permission checks",
    )
    def test_unreadable_file_raises_value_error(self, tmp_path):
        # A file with an unrecognised extension forces the content-sniff path.
        unreadable = tmp_path / "data.sacc"
        unreadable.write_bytes(b"irrelevant")
        unreadable.chmod(0o000)          # remove all permissions

        try:
            with pytest.raises(ValueError, match="Could not detect file type"):
                detect_sacc_file_type(str(unreadable))
        finally:
            unreadable.chmod(stat.S_IRUSR | stat.S_IWUSR)  # restore so tmp_path cleanup works

    # --- sanity: extension-based detection still works (regression guard) --

    def test_fits_gz_extension(self, tmp_path):
        assert detect_sacc_file_type("something.fits.gz") == "fits"

    def test_hdf5_gz_extension(self, tmp_path):
        assert detect_sacc_file_type("something.hdf5.gz") == "hdf5"

    def test_fits_extension(self, tmp_path):
        assert detect_sacc_file_type("something.fits") == "fits"

    def test_hdf5_extension(self, tmp_path):
        assert detect_sacc_file_type("something.hdf5") == "hdf5"
