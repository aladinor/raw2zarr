import gzip
import os
from unittest.mock import patch

from raw2zarr.io.preprocess import normalize_input_for_xradar


def test_local_uncompressed_file(tmp_path):
    f = tmp_path / "radar.vol"
    f.write_text("test-data")
    out = normalize_input_for_xradar(str(f))
    assert out == str(f)


def test_local_gz_file(tmp_path):
    gz_file = tmp_path / "radar.vol.gz"
    raw_data = b"test-gz-content"
    with gzip.open(gz_file, "wb") as f:
        f.write(raw_data)

    out_path = normalize_input_for_xradar(str(gz_file))
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == raw_data
    os.remove(out_path)


@patch("raw2zarr.io.preprocess.fsspec.open_local")
def test_s3_uncompressed_file(mock_open_local, tmp_path):
    f = tmp_path / "downloaded.vol"
    f.write_text("s3 test data")
    mock_open_local.return_value = str(f)

    s3_path = "s3://fake-bucket/file.vol"
    out = normalize_input_for_xradar(s3_path)

    assert mock_open_local.called
    assert out == str(f)


@patch("raw2zarr.io.preprocess.fsspec.open_local")
def test_s3_gz_file(mock_open_local, tmp_path):
    gz_file = tmp_path / "remote.gz"
    raw_data = b"gz-data-from-s3"

    with gzip.open(gz_file, "wb") as f:
        f.write(raw_data)

    mock_open_local.return_value = str(gz_file)
    s3_gz_path = "s3://fake-bucket/file.gz"
    out = normalize_input_for_xradar(s3_gz_path)

    assert mock_open_local.called
    assert os.path.exists(out)
    with open(out, "rb") as f:
        assert f.read() == raw_data
    os.remove(out)
