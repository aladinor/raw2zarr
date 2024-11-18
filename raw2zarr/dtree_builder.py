from typing import Iterable, List, Union
import os

from xarray import DataTree
from xarray.backends.common import _normalize_path

# Relative imports
from .data_reader import accessor_wrapper
from .utils import ensure_dimension, fix_angle, dtree_encoding, batch


def datatree_builder(
    filename_or_obj: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
    engine: str = "iris",
    dim: str = "vcp_time",
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
        dim (str, optional):
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
        >>> tree = datatree_builder(["file1.RAW", "file2.RAW"], engine="iris", dim="vcp_time")
        >>> print(tree)
        >>> print(tree["root/child"].to_dataset())  # Access a node's dataset
    """
    # Initialize an empty dictionary to hold the nested structure

    # Load radar data in batches
    filename_or_obj = _normalize_path(filename_or_obj)
    dtree = accessor_wrapper(filename_or_obj, engine=engine)
    task_name = dtree.attrs.get("scan_name", "default_task").strip()
    dtree = (dtree.pipe(fix_angle).pipe(ensure_dimension, dim)).xradar.georeference()
    dtree = DataTree.from_dict({task_name: dtree})
    dtree.encoding = dtree_encoding(dtree, append_dim=dim)
    return dtree


def process_file(file: str, engine: str = "nexradlevel2") -> DataTree:
    """
    Load and transform a single radar file into a DataTree object.
    """
    try:
        dtree = datatree_builder(file, engine=engine)
        return dtree
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None


def append_sequential(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    zarr_store: str,
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

    Example:
        Process a list of radar files sequentially and append them to a Zarr store:

        >>> radar_files = ["file1.RAW", "file2.RAW", "file3.RAW"]
        >>> zarr_store = "output.zarr"
        >>> append_sequential(
        ...     radar_files=radar_files,
        ...     append_dim="vcp_time",
        ...     zarr_store=zarr_store,
        ...     engine="iris",
        ...     zarr_format=2
        ... )
    """
    for file in radar_files:
        dtree = process_file(file, engine=engine)
        zarr_format = kwargs.get("zarr_format", 2)
        if dtree:
            enc = dtree.encoding
            dtree = dtree[dtree.groups[1]]
            try:
                dtree.to_zarr(
                    store=zarr_store,
                    mode="a-",
                    encoding=enc,
                    consolidated=True,
                    zarr_format=zarr_format,
                )
            except ValueError:
                dtree.to_zarr(
                    store=zarr_store,
                    mode="a-",
                    consolidated=True,
                    append_dim=append_dim,
                    zarr_format=zarr_format,
                )
    print("done")


def append_parallel(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    zarr_store: str,
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

    Example:
        >>> radar_files = ["file1.nc", "file2.nc", "file3.nc"]
        >>> zarr_store = "output.zarr"
        >>> append_parallel(
                radar_files=radar_files,
                append_dim="time",
                zarr_store=zarr_store,
                engine="nexradlevel2",
                batch_size=5
            )
    """

    from functools import partial
    from dask import bag as db
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(dashboard_address="127.0.0.1:8785")
    client = Client(cluster)
    pf = partial(process_file, engine=engine)

    if not batch_size:
        batch_size = sum(client.ncores().values())

    for files in batch(radar_files, n=batch_size):
        bag = db.from_sequence(files, npartitions=len(files)).map(pf)
        ls_dtree: List[DataTree] = bag.compute()
        for dtree in ls_dtree:
            zarr_format = kwargs.get("zarr_format", 2)
            if dtree:
                enc = dtree.encoding
                dtree = dtree[dtree.groups[1]]
                try:
                    dtree.to_zarr(
                        store=zarr_store,
                        mode="a-",
                        encoding=enc,
                        consolidated=True,
                        zarr_format=zarr_format,
                    )
                except ValueError:
                    dtree.to_zarr(
                        store=zarr_store,
                        mode="a-",
                        consolidated=True,
                        append_dim=append_dim,
                        zarr_format=zarr_format,
                    )
