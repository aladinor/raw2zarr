import bz2
import gzip
from typing import IO, Union

import fsspec


def prepare2read(
    filename: Union[str, IO],
    storage_options: dict = None,
) -> IO:
    """
    Return a file-like object ready for streaming read.
    Supports:
      - Local files
      - S3 objects
      - GZIP and BZIP2 compression
    Use only for formats that accept file-like input (e.g., IRIS, ODIM).
    """
    if not storage_options:
        storage_options = {"anon": True}

    if hasattr(filename, "read"):
        return filename

    if isinstance(filename, str) and filename.startswith("s3://"):
        stream = fsspec.open(
            filename, mode="rb", compression="infer", **storage_options
        ).open()
        return stream.read()

    file = open(filename, "rb")
    magic = file.read(3)
    file.seek(0)

    if magic.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(fileobj=file)
    elif magic.startswith(b"BZh"):
        return bz2.BZ2File(file)

    return file
