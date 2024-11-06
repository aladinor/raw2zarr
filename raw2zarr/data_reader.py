from typing import List, Iterable
import os
import xradar
import dask.bag as db
from xarray import DataTree
from xarray.backends.common  import _normalize_path
from raw2zarr.utils import prepare_for_read, batch


def accessor_wrapper(
        filename_or_obj: str | os.PathLike | Iterable[str | os.PathLike],
        backend: str = "iris"
) -> DataTree:
    """Wrapper function to load radar data for a single file or iterable of files with fsspec and compression check."""
    try:
        if isinstance(filename_or_obj, Iterable) and not isinstance(filename_or_obj, (str, os.PathLike)):
            results = []
            for file in filename_or_obj:
                prepared_file = prepare_for_read(file)
                results.append(_load_file(prepared_file, backend))
            return results
        else:
            file = prepare_for_read(filename_or_obj)
            return _load_file(file, backend)
    except Exception as e:
        print(f"Error loading {filename_or_obj}: {e}")
        return None


def _load_file(file, backend) -> DataTree:
    """Helper function to load a single file with the specified backend."""
    if backend == "iris":
        return xradar.io.open_iris_datatree(file)
    elif backend == "odim":
        return xradar.io.open_odim_datatree(file)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def load_radar_data(
        filename_or_obj: str | os.PathLike | Iterable[str | os.PathLike],
        backend: str = "iris",
        parallel: bool = False,
        batch_size: int = 12
) -> Iterable[List[DataTree]]:
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
            bag = db.from_sequence(files_batch, npartitions=len(files_batch)).map(accessor_wrapper, backend=backend)
            ls_dtree.extend(bag.compute())
        else:
            for file_path in files_batch:
                result = accessor_wrapper(file_path, backend=backend)
                if result is not None:
                    ls_dtree.append(result)

        yield ls_dtree  # Yield each batch of DataTree objects
