from __future__ import annotations

import os
from collections.abc import Iterable

from xarray import DataTree
from xarray.backends.common import _normalize_path

from raw2zarr.writer.zarr_writer import dtree2zarr

# Relative imports
from ..io.load import load_radar_data
from ..transform.alignment import align_dynamic_scan, check_dynamic_scan, fix_angle
from ..transform.dimension import ensure_dimension
from ..transform.encoding import dtree_encoding
from ..transform.georeferencing import apply_georeferencing
from ..utils import (
    batch,
)


def datatree_builder(
    filename_or_obj: str | os.PathLike | Iterable[str | os.PathLike],
    engine: str = "iris",
    append_dim: str = "vcp_time",
) -> DataTree:
    """
    Construct a hierarchical xarray.DataTree from radar data files.

    This function loads radar data from one or more files and organizes it into a nested
    `xarray.DataTree` structure. The data can be processed in batches and supports different
    backend engines for reading the data.

    Parameters:
        filename_or_obj (str | os.PathLike | Iterable[str | os.PathLike]):
            Path or paths to the radar data files to be loaded. Can be a single file,
            a directory path, or an iterable of file paths.
        engine (str, optional):
            The backend engine to use for loading the radar data. Common options include
            'iris' (default) and 'odim'. The selected engine must be supported by the underlying
            data processing libraries.
        append_dim (str, optional):
            The name of the dimension to use for concatenating data across files. Default is 'vcp_time'.
            Note: The 'time' dimension cannot be used as the concatenation dimension because it is
            already a predefined dimension in the dataset and reserved for temporal data. Choose
            a unique dimension name that does not conflict with existing dimensions in the datasets.


    Returns:
        xarray.DataTree:
            A nested `xarray.DataTree` object that combines all the loaded radar data files into a
            hierarchical structure. Each node in the tree corresponds to an `xarray.Dataset`.

    Raises:
        ValueError:
            If no files are successfully loaded or if all batches result in empty data.

    Notes:
        - This function is designed to handle large datasets efficiently, potentially
          processing data in batches and leveraging parallelism if supported by the backend.
        - The resulting `xarray.DataTree` retains a hierarchical organization based on the structure
          of the input files and their metadata.

    Example:
        >>> from raw2zarr import datatree_builder
        >>> tree = datatree_builder(["file1.RAW", "file2.RAW"], engine="iris", append_dim="vcp_time")
        >>> print(tree)
        >>> print(tree["root/child"].to_dataset())  # Access a node's dataset
    """

    filename_or_obj = _normalize_path(filename_or_obj)
    dtree = load_radar_data(filename_or_obj, engine=engine)
    task_name = dtree.attrs.get("scan_name", "default_task").strip()
    dtree = dtree.pipe(fix_angle).pipe(apply_georeferencing)
    if (engine == "nexradlevel2") & check_dynamic_scan(dtree):
        dtree = align_dynamic_scan(dtree, append_dim=append_dim)
    else:
        dtree = dtree.pipe(ensure_dimension, append_dim)
    dtree = DataTree.from_dict({task_name: dtree})
    dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)
    return dtree


def process_file(file: str, engine: str = "nexradlevel2") -> DataTree:
    """
    Load and transform a single radar file into a DataTree object.
    """
    try:
        return datatree_builder(file, engine=engine)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None


def append_sequential(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    zarr_store: str,
    zarr_format: int = 3,
    consolidated: bool = False,
    engine: str = "iris",
    **kwargs,
) -> None:
    """
    Sequentially processes radar files and appends their data to a Zarr store.

    This function processes radar files one at a time, converting each file into an
    `xarray.DataTree` object and sequentially appending its data to the specified Zarr store.
    The process ensures data is written in an ordered manner along the specified dimension.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            An iterable containing file paths to the radar data files to be processed.
        append_dim (str):
            The dimension along which data is appended in the Zarr store. Typically used
            to represent temporal or scan-specific dimensions (e.g., "vcp_time").
        zarr_store (str):
            The file path or URL to the output Zarr store where data will be appended.
        engine (str, optional):
            The backend engine to use for loading radar data. Options include:
            - "iris" (default): For IRIS format radar data.
            - "nexradlevel2": For NEXRAD Level 2 data.
            - "odim": For ODIM HDF5 format.
        **kwargs:
            Additional optional parameters, including:
            - zarr_format (int, optional): The Zarr format version to use (default: 2).

    Returns:
        None:
            The function does not return any values. Processed radar data is written
            directly to the specified Zarr store.

    Raises:
        ValueError:
            If an error occurs during appending to the Zarr store or if the provided
            dimension or file paths are invalid.

    Notes:
        - Data is written sequentially to the Zarr store, ensuring an ordered structure
          along the specified `append_dim`.
        - Handles encoding for compatibility with the Zarr format, including time and
          custom dimension variables.
        - Supports customization via the `engine` parameter for different radar data formats.
    """
    for file in radar_files:
        print(file)
        dtree = process_file(file, engine=engine)
        if dtree:
            enc = dtree.encoding
            dtree2zarr(
                dtree,
                store=zarr_store,
                mode="a-",
                encoding=enc,
                consolidated=consolidated,
                zarr_format=zarr_format,
                append_dim=append_dim,
                write_inherited_coords=True,
            )


def append_parallel(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    zarr_store: str,
    zarr_format: int = 3,
    consolidated: bool = False,
    engine: str = "nexradlevel2",
    batch_size: int = None,
    **kwargs,
) -> None:
    """
    Load radar files in parallel and append their data sequentially to a Zarr store.

    This function uses Dask Bag to load radar files in parallel, processing them in
    configurable batches. After loading, the resulting `xarray.DataTree` objects are
    processed and written sequentially to the Zarr store, ensuring consistent and ordered
    data storage. A Dask LocalCluster is used to distribute computation across available cores.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            An iterable containing paths to the radar files to process.
        append_dim (str):
            The dimension along which to append data in the Zarr store.
        zarr_store (str):
            The path to the output Zarr store where data will be written.
        engine (str, optional):
            The backend engine used to load radar files. Defaults to "nexradlevel2".
        batch_size (int, optional):
            The number of files to process in each batch. If not specified, it defaults to
            the total number of cores available in the Dask cluster.
        **kwargs:
            Additional arguments, including:
                - zarr_format (int, optional): The Zarr format version to use (default: 2).

    Returns:
        None:
            This function writes data directly to the specified Zarr store and does not return a value.

    Notes:
        - File loading is parallelized using Dask Bag for efficiency, but data writing
          to the Zarr store is performed sequentially to ensure consistent and ordered output.
        - A Dask LocalCluster is created with a web-based dashboard for monitoring at
          `http://127.0.0.1:8785` by default.
        - If `batch_size` is not specified, it is automatically set based on the available cores
          in the Dask cluster.
    """

    import gc
    from functools import partial

    from dask import bag as db
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(dashboard_address="127.0.0.1:8785", memory_limit="10GB")
    client = Client(cluster)
    pf = partial(process_file, engine=engine)

    if not batch_size:
        batch_size = sum(client.ncores().values())

    for radar_files_batch in batch(radar_files, n=batch_size):
        bag = db.from_sequence(radar_files_batch, npartitions=batch_size).map(pf)
        ls_dtree: list[DataTree] = bag.compute()
        for dtree in ls_dtree:
            if dtree:
                dtree2zarr(
                    dtree,
                    store=zarr_store,
                    mode="a-",
                    encoding=dtree.encoding,
                    consolidated=consolidated,
                    zarr_format=zarr_format,
                    append_dim=append_dim,
                    compute=True,
                    write_inherited_coords=True,
                )
        del bag, ls_dtree
        gc.collect()
