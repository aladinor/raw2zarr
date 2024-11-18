from typing import List, Iterable
import os

import xradar
import fsspec
from xarray import DataTree
from s3fs import S3File

# Relative imports
from .utils import prepare_for_read, batch


def accessor_wrapper(
    filename_or_obj: str | os.PathLike,
    engine: str = "iris",
) -> DataTree:
    """Wrapper function to load radar data for a single file or iterable of files with fsspec and compression check."""
    try:
        file = prepare_for_read(filename_or_obj)
        return _load_file(file, engine)
    except Exception as e:
        print(f"Error loading {filename_or_obj}: {e}")
        return None


def _load_file(file, engine) -> DataTree:
    """Helper function to load a single file with the specified backend."""
    if engine == "iris":
        if isinstance(file, S3File):
            return xradar.io.open_iris_datatree(file.read())
        elif isinstance(file, bytes):
            return xradar.io.open_iris_datatree(file)
        else:
            return xradar.io.open_iris_datatree(file)
    elif engine == "odim":
        return xradar.io.open_odim_datatree(file)
    elif engine == "nexradlevel2":
        if isinstance(file, S3File):
            local_file = fsspec.open_local(
                f"simplecache::s3://{file.path}",
                s3={"anon": True},
                filecache={"cache_storage": "."},
            )
            data_tree = xradar.io.open_nexradlevel2_datatree(local_file)

            # Remove the local file after loading the data
            os.remove(local_file)
            return data_tree
        else:
            return xradar.io.open_nexradlevel2_datatree(file)
    else:
        raise ValueError(f"Unsupported backend: {engine}")
