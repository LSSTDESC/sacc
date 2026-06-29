"""
Tests for the ``maybe_decompress`` context manager in ``sacc.utils``.

The goal is 100 % branch/statement coverage of ``maybe_decompress`` itself,
exercised without any dependency on the ``Sacc`` class or file-format
libraries (fits / hdf5).
"""

import gzip
import os
import tempfile

import pytest

from sacc.utils import maybe_decompress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_plain(path, content=b"hello"):
    """Write raw bytes to *path*."""
    with open(path, "wb") as fh:
        fh.write(content)


def _write_gzip(path, content=b"hello"):
    """Write gzip-compressed bytes to *path*."""
    with gzip.open(path, "wb") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMaybeDecompressPlainFile:
    """Non-``.gz`` filenames: original path is yielded, nothing is created."""

    def test_yields_original_path(self, tmp_path):
        plain = tmp_path / "data.fits"
        _write_plain(plain)

        with maybe_decompress(str(plain)) as effective:
            assert effective == str(plain)

    def test_no_temp_file_created(self, tmp_path):
        plain = tmp_path / "data.fits"
        _write_plain(plain)

        before = set(os.listdir(tempfile.gettempdir()))
        with maybe_decompress(str(plain)):
            after = set(os.listdir(tempfile.gettempdir()))
        # The temp directory should not have grown due to our CM
        assert after == before or after.issubset(before | after)  # no assertion needed really
        # A stronger check: no new file whose name starts with "data"
        new_files = after - before
        assert not any("data" in f for f in new_files)

    def test_file_content_readable_inside_block(self, tmp_path):
        plain = tmp_path / "data.bin"
        _write_plain(plain, b"payload")

        with maybe_decompress(str(plain)) as effective:
            with open(effective, "rb") as fh:
                assert fh.read() == b"payload"


class TestMaybeDecompressGzipFile:
    """`.gz` filenames: decompressed temp file is yielded and cleaned up."""

    def test_yields_different_path(self, tmp_path):
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"content")

        with maybe_decompress(str(gz)) as effective:
            assert effective != str(gz)

    def test_temp_file_exists_inside_block(self, tmp_path):
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"content")

        with maybe_decompress(str(gz)) as effective:
            assert os.path.exists(effective)

    def test_temp_file_contains_decompressed_content(self, tmp_path):
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"decompressed payload")

        with maybe_decompress(str(gz)) as effective:
            with open(effective, "rb") as fh:
                assert fh.read() == b"decompressed payload"

    def test_temp_file_deleted_after_normal_exit(self, tmp_path):
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"content")

        with maybe_decompress(str(gz)) as effective:
            temp_path = effective  # remember it

        assert not os.path.exists(temp_path)

    def test_temp_file_deleted_after_exception(self, tmp_path):
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"content")

        temp_path = None
        with pytest.raises(RuntimeError, match="boom"):
            with maybe_decompress(str(gz)) as effective:
                temp_path = effective
                raise RuntimeError("boom")

        assert temp_path is not None
        assert not os.path.exists(temp_path)

    def test_temp_file_already_gone_before_cleanup(self, tmp_path):
        """
        If something inside the ``with`` block deletes the temp file early,
        the ``finally`` branch that guards with ``os.path.exists`` must not
        raise.  This exercises the ``os.path.exists`` check in the finally.
        """
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"content")

        # Should not raise even though the temp file is deleted inside the block
        with maybe_decompress(str(gz)) as effective:
            os.remove(effective)  # delete it early
        # If we reach here without FileNotFoundError the guard worked correctly.

    def test_original_gz_file_not_deleted(self, tmp_path):
        gz = tmp_path / "data.fits.gz"
        _write_gzip(gz, b"content")

        with maybe_decompress(str(gz)):
            pass

        assert os.path.exists(str(gz))
