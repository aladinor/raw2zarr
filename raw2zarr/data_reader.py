from typing import List, Iterable
import os

import xradar
import fsspec
import dask.bag as db
from xarray import DataTree
from xarray.backends.common import _normalize_path
from s3fs import S3File

# Relative imports
from .utils import prepare_for_read, batch, fix_angle


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


def _process_file(args):
    file, engine = args
    return accessor_wrapper(file, engine=engine)


def load_radar_data(
    filename_or_obj: str | os.PathLike | Iterable[str | os.PathLike],
    backend: str = "iris",
    parallel: bool = False,
    batch_size: int = 12,
) -> DataTree:
    """
    Load radar data from files in batches to avoid memory overload.

    Parameters:
        filename_or_obj (str | os.PathLike | Iterable[str | os.PathLike]): Path(s) to radar data files.
        backend (str): Backend type to use. Options include 'iris', 'odim', etc. Default is 'iris'.
        parallel (bool): If True, enables parallel processing with Dask. Default is False.
        batch_size (int): Number of files to process in each batch.

    Returns:
        Iterable[List[DataTree]]: An iterable yielding batches of DataTree objects.
    """
    filename_or_obj = _normalize_path(filename_or_obj)

    for files_batch in batch(filename_or_obj, batch_size):
        ls_dtree = []

        if parallel:
            bag = db.from_sequence(files_batch, npartitions=len(files_batch)).map(
                accessor_wrapper, backend=backend
            )
            ls_dtree.extend(bag.compute())
        else:
            for file_path in files_batch:
                result = accessor_wrapper(file_path, engine=backend)
                if result is not None:
                    ls_dtree.append(result)

        yield ls_dtree  # Yield each batch of DataTree objects
