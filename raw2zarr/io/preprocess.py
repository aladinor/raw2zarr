import gzip
import tempfile
from typing import Union

import fsspec
from s3fs.core import S3File


def normalize_input_for_xradar(
    file: Union[str, S3File], storage_options: dict = None
) -> Union[str, bytes]:
    """
    Prepares radar data for xradar loaders.

    Automatically detects local vs S3 files:
    - S3 files (s3://): Uses streaming with automatic gzip decompression
    - Local files: Returns file path, decompresses .gz to temp file if needed

    Parameters:
        file (str or S3File): Path or object pointing to radar data.
        storage_options (dict, optional): Additional storage options for fsspec

    Returns:
        Union[str, bytes]: Local file path (for local files) or bytes data (for S3 files)
    """
    if storage_options is None:
        storage_options = {}

    is_remote = isinstance(file, S3File) or (
        isinstance(file, str) and file.startswith("s3://")
    )
    is_gz = isinstance(file, str) and file.endswith(".gz")

    # Use streaming for remote files
    if is_remote and isinstance(file, str):
        # Determine compression from file extension
        compression = None
        if is_gz:
            compression = "gzip"
        elif file.endswith(".bz2"):
            compression = "bz2"

        # Set default anonymous access for S3
        if file.startswith("s3://") and "anon" not in storage_options:
            storage_options["anon"] = True

        try:
            # Stream the data directly
            with fsspec.open(
                file, mode="rb", compression=compression, **storage_options
            ) as f:
                return f.read()
        except Exception as e:
            print(
                f"[Warning] Streaming failed for {file}: {e}. Falling back to local processing."
            )
            # Fall through to original logic

    # Handle S3File objects or fallback for remote files
    if isinstance(file, S3File) or (isinstance(file, str) and file.startswith("s3://")):
        s3_path = f"simplecache::{file.path if isinstance(file, S3File) else file}"
        local_file = fsspec.open_local(
            s3_path, s3={"anon": True}, filecache={"cache_storage": "."}
        )
        if file.endswith(".gz"):
            return _decompress_to_temp(local_file)
        return local_file

    # Local files - decompress .gz if needed
    elif is_gz:
        return _decompress_to_temp(file)

    # Local uncompressed files
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
