import gzip
import tempfile
import fsspec
from s3fs.core import S3File
from typing import Union


def normalize_input_for_xradar(file: Union[str, S3File]) -> str:
    """
    Ensures a local, uncompressed file path suitable for xradar loaders (e.g., NEXRAD).

    Handles:
    - S3 files (via fsspec and simplecache)
    - .gz compression (via temporary decompression)
    - Local uncompressed files (passed as-is)

    Parameters:
        file (str or S3File): Path or object pointing to radar data.

    Returns:
        str: Local file path, guaranteed to be uncompressed.
    """
    is_s3 = isinstance(file, S3File) or (
        isinstance(file, str) and file.startswith("s3://")
    )
    is_gz = isinstance(file, str) and file.endswith(".gz")

    if is_s3:
        s3_path = f"simplecache::{file.path if isinstance(file, S3File) else file}"
        local_file = fsspec.open_local(
            s3_path, s3={"anon": True}, filecache={"cache_storage": "."}
        )
        if file.endswith(".gz"):
            return _decompress_to_temp(local_file)
        return local_file

    elif is_gz:
        return _decompress_to_temp(file)

    return file


def _decompress_to_temp(gz_path: str) -> str:
    """
    Decompress a GZIP file to a temporary file.

    Parameters:
        gz_path (str): Path to the gzip file

    Returns:
        str: Path to a temporary uncompressed file
    """
    with (
        gzip.open(gz_path, "rb") as gz,
        tempfile.NamedTemporaryFile(delete=False, suffix=".nexrad") as tmp,
    ):
        tmp.write(gz.read())
        return tmp.name
