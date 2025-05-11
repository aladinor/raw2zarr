import gzip
import bz2
import io
from unittest.mock import patch, MagicMock

from raw2zarr.io.base import prepare2read


def test_prepare2read_local_plain_file(tmp_path):
    file = tmp_path / "test.vol"
    file.write_text("plain radar data")
    with prepare2read(str(file)) as f:
        assert f.read() == b"plain radar data"


def test_prepare2read_local_gz_file(tmp_path):
    path = tmp_path / "test.vol.gz"
    content = b"gzip radar"
    with gzip.open(path, "wb") as f:
        f.write(content)

    with prepare2read(str(path)) as f:
        assert f.read() == content


def test_prepare2read_local_bz2_file(tmp_path):
    path = tmp_path / "test.vol.bz2"
    content = b"bz2 radar"
    with bz2.BZ2File(path, "wb") as f:
        f.write(content)

    with prepare2read(str(path)) as f:
        assert f.read() == content


def test_prepare2read_filelike():
    buffer = io.BytesIO(b"in-memory stream")
    out = prepare2read(buffer)
    assert out is buffer
    assert out.read() == b"in-memory stream"


@patch("raw2zarr.io.base.fsspec.open")
def test_prepare2read_s3_file(mock_fsspec_open):
    content = b"s3 content"
    fake_stream = io.BytesIO(content)
    mock_open_context = MagicMock()
    mock_open_context.open.return_value = fake_stream
    mock_fsspec_open.return_value = mock_open_context

    s3_path = "s3://bucket/somefile.vol"
    out = prepare2read(s3_path)

    mock_fsspec_open.assert_called_once()
    assert out.read() == content
