import gzip
import os
from unittest.mock import MagicMock, patch

from raw2zarr.io.preprocess import normalize_input_for_xradar


class TestNormalizeInput:
    def test_local_uncompressed_file(self, tmp_path):
        """Test local uncompressed files return file path unchanged."""
        f = tmp_path / "radar.vol"
        f.write_text("test-data")
        out = normalize_input_for_xradar(str(f))
        assert out == str(f)
        assert isinstance(out, str)

    def test_local_gz_file(self, tmp_path):
        """Test local .gz files are decompressed to temporary files."""
        gz_file = tmp_path / "radar.vol.gz"
        raw_data = b"test-gz-content"
        with gzip.open(gz_file, "wb") as f:
            f.write(raw_data)

        out_path = normalize_input_for_xradar(str(gz_file))
        assert isinstance(out_path, str)
        assert os.path.exists(out_path)
        with open(out_path, "rb") as f:
            assert f.read() == raw_data
        os.remove(out_path)

    @patch("raw2zarr.io.preprocess.fsspec.open")
    def test_s3_streaming_uncompressed(self, mock_fsspec_open):
        """Test S3 uncompressed files are streamed and return bytes."""
        test_data = b"s3 test data content"
        mock_file = MagicMock()
        mock_file.read.return_value = test_data
        mock_fsspec_open.return_value.__enter__.return_value = mock_file

        s3_path = "s3://fake-bucket/file.vol"
        out = normalize_input_for_xradar(s3_path)

        # Should stream and return bytes
        assert isinstance(out, bytes)
        assert out == test_data
        mock_fsspec_open.assert_called_once_with(
            s3_path, mode="rb", compression=None, anon=True
        )

    @patch("raw2zarr.io.preprocess.fsspec.open")
    def test_s3_streaming_gzipped(self, mock_fsspec_open):
        """Test S3 .gz files are streamed with automatic decompression."""
        test_data = b"gzipped s3 content"
        mock_file = MagicMock()
        mock_file.read.return_value = test_data
        mock_fsspec_open.return_value.__enter__.return_value = mock_file

        s3_gz_path = "s3://fake-bucket/file.gz"
        out = normalize_input_for_xradar(s3_gz_path)

        # Should stream with gzip decompression and return bytes
        assert isinstance(out, bytes)
        assert out == test_data
        mock_fsspec_open.assert_called_once_with(
            s3_gz_path, mode="rb", compression="gzip", anon=True
        )

    @patch("raw2zarr.io.preprocess.fsspec.open")
    def test_s3_streaming_with_storage_options(self, mock_fsspec_open):
        """Test S3 streaming respects custom storage options."""
        test_data = b"custom options test"
        mock_file = MagicMock()
        mock_file.read.return_value = test_data
        mock_fsspec_open.return_value.__enter__.return_value = mock_file

        s3_path = "s3://private-bucket/file.vol"
        storage_options = {"anon": False, "key": "access_key"}
        out = normalize_input_for_xradar(s3_path, storage_options=storage_options)

        assert isinstance(out, bytes)
        assert out == test_data
        mock_fsspec_open.assert_called_once_with(
            s3_path, mode="rb", compression=None, anon=False, key="access_key"
        )

    @patch("raw2zarr.io.preprocess.fsspec.open_local")
    @patch("raw2zarr.io.preprocess.fsspec.open")
    def test_s3_streaming_fallback(self, mock_fsspec_open, mock_open_local, tmp_path):
        """Test fallback to local caching when streaming fails."""
        # Make streaming fail
        mock_fsspec_open.side_effect = Exception("Streaming failed")

        # Setup fallback local file
        f = tmp_path / "fallback.vol"
        f.write_text("fallback data")
        mock_open_local.return_value = str(f)

        s3_path = "s3://fake-bucket/file.vol"
        out = normalize_input_for_xradar(s3_path)

        # Should fallback and return file path
        assert isinstance(out, str)
        assert out == str(f)
        mock_fsspec_open.assert_called_once()
        mock_open_local.assert_called_once()

    @patch("raw2zarr.io.preprocess.fsspec.open_local")
    @patch("raw2zarr.io.preprocess.fsspec.open")
    def test_s3_gz_streaming_fallback(
        self, mock_fsspec_open, mock_open_local, tmp_path
    ):
        """Test fallback for .gz files when streaming fails."""
        # Make streaming fail
        mock_fsspec_open.side_effect = Exception("Streaming failed")

        # Setup fallback gz file
        gz_file = tmp_path / "fallback.gz"
        raw_data = b"fallback gz data"
        with gzip.open(gz_file, "wb") as f:
            f.write(raw_data)
        mock_open_local.return_value = str(gz_file)

        s3_gz_path = "s3://fake-bucket/file.gz"
        out = normalize_input_for_xradar(s3_gz_path)

        # Should fallback, decompress, and return temp file path
        assert isinstance(out, str)
        assert os.path.exists(out)
        with open(out, "rb") as f:
            assert f.read() == raw_data
        mock_fsspec_open.assert_called_once()
        mock_open_local.assert_called_once()
        os.remove(out)
