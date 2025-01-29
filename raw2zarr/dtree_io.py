from __future__ import annotations

import gzip
import logging
import os
import tempfile
from typing import Callable

import fsspec
import xradar as xd
from s3fs.core import S3File
from xarray import DataTree

# Relative imports
from .utils import prepare2read

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = {"iris", "nexradlevel2"}

ENGINE_REGISTRY: dict[str, Callable] = {}


def iris_loader(file: str | bytes | S3File) -> DataTree:
    """
    Loads iris files
    """
    if isinstance(file, (S3File, bytes)):
        return xd.io.open_iris_datatree(
            file.read() if isinstance(file, S3File) else file
        )
    return xd.io.open_iris_datatree(file)


def odim_loader(file: str) -> DataTree:
    """
    Loads odim files
    """
    return xd.io.open_odim_datatree(file)


def nexradlevel2_loader(filename_or_obj: str | bytes | S3File) -> DataTree:
    """
    Loads Nexrad level2 files
    """
    if isinstance(filename_or_obj, S3File):
        # As xradar.io.open_nexradlevel2_datatree does not support data streaming we need to download the file
        local_file = fsspec.open_local(
            f"simplecache::s3://{filename_or_obj.path}",
            s3={"anon": True},
            filecache={"cache_storage": "."},
        )
        try:
            if filename_or_obj.path.endswith(".gz"):
                return _decompress(local_file)
            return xd.io.open_nexradlevel2_datatree(local_file)
        finally:
            os.remove(local_file)
    elif isinstance(filename_or_obj, gzip.GzipFile):
        try:
            local_file = fsspec.open_local(
                f"simplecache::s3://{filename_or_obj.fileobj.full_name}",
                s3={"anon": True},
                filecache={"cache_storage": "."},
            )
            return xd.io.open_nexradlevel2_datatree(_decompress(local_file))
        finally:
            os.remove(local_file)
    elif isinstance(filename_or_obj, str):
        if not filename_or_obj.startswith("s3://") and filename_or_obj.endswith(".gz"):
            return xd.io.open_nexradlevel2_datatree(_decompress(filename_or_obj))
        elif filename_or_obj.startswith("s3://") and not filename_or_obj.endswith(
            ".gz"
        ):
            local_file = fsspec.open_local(
                f"simplecache::s3://{filename_or_obj}",
                s3={"anon": True},
                filecache={"cache_storage": "."},
            )
            return xd.io.open_nexradlevel2_datatree(local_file)
        elif filename_or_obj.startswith("s3://") and filename_or_obj.endswith(".gz"):
            local_file = fsspec.open_local(
                f"simplecache::s3://{filename_or_obj}",
                s3={"anon": True},
                filecache={"cache_storage": "."},
            )
            return xd.io.open_nexradlevel2_datatree(_decompress(local_file))
    return xd.io.open_nexradlevel2_datatree(filename_or_obj)


def _decompress(gz_file: str) -> DataTree:
    """
    Decompress a GZIP file and load it into a DataTree.

    Parameters
    ----------
    gz_file : str
        Path to the GZIP-compressed file.

    Returns
    -------
    DataTree
        The loaded radar data.
    """
    with (
        gzip.open(gz_file, "rb") as gz,
        tempfile.NamedTemporaryFile(delete=False) as temp_file,
    ):
        temp_file.write(gz.read())
        temp_file_path = temp_file.name
    return temp_file_path


ENGINE_REGISTRY["iris"] = iris_loader
ENGINE_REGISTRY["odim"] = odim_loader
ENGINE_REGISTRY["nexradlevel2"] = nexradlevel2_loader


def _load_file(filename: str | bytes | S3File, engine: str) -> DataTree:
    """Helper function to load a single file with the specified backend."""
    if engine not in ENGINE_REGISTRY:
        raise ValueError(f"Unsupported backend: {engine}")
    backend_function = ENGINE_REGISTRY[engine]
    return backend_function(filename)


def load_radar_data(
    filename_or_obj: str | os.PathLike | bytes | S3File,
    engine: str = "iris",
) -> DataTree:
    """Wrapper function to load radar data for a single file or iterable of files with fsspec and compression check."""
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unsupported engine: {engine}. Must be one of {SUPPORTED_ENGINES}"
        )
    try:
        file = prepare2read(filename_or_obj)
        return _load_file(file, engine)
    except Exception as e:
        logger.error(f"Failed to load {filename_or_obj} with engine {engine}: {e}")
        return None
