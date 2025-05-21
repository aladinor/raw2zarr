from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Literal

from xarray import DataTree

from ..utils import batch
from ..writer.zarr_writer import dtree2zarr
from .dtree_radar import radar_datatree


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
    Append radar files to a Zarr store sequentially.

    This function loads each radar file one at a time, converts it to a DataTree,
    and appends it to the specified Zarr store. It is suitable for smaller datasets
    or environments where parallel processing is not needed.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            Paths to the radar data files to be appended.
        append_dim (str):
            Name of the dimension along which data should be appended (e.g., "vcp_time").
        zarr_store (str):
            Path to the output Zarr store.
        zarr_format (int, optional):
            Zarr format version to use (2 or 3). Defaults to 3.
        consolidated (bool, optional):
            Whether to consolidate Zarr metadata. Defaults to False.
        engine (str, optional):
            Radar file reading engine (e.g., "iris", "nexradlevel2"). Defaults to "iris".
        **kwargs:
            Additional options passed to the Zarr writer.

    Returns:
        None
    """
    for file in radar_files:
        try:
            dtree = radar_datatree(file, engine=engine)
        except Exception as e:
            print(f"[Warning] Failed to process {file}: {e}")
            continue

        if dtree:
            dtree2zarr(
                dtree,
                store=zarr_store,
                mode="a-",
                encoding=dtree.encoding,
                consolidated=consolidated,
                zarr_format=zarr_format,
                append_dim=append_dim,
                write_inherited_coords=True,
                **kwargs,
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
    Append radar files to a Zarr store in parallel using Dask.

    This function processes radar files in parallel using Dask, converts them to
    xarray.DataTree objects, and sequentially appends them to the specified Zarr store.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            Iterable of radar file paths to process.
        append_dim (str):
            Name of the dimension along which to append (e.g., "vcp_time").
        zarr_store (str):
            Path to the output Zarr store.
        zarr_format (int, optional):
            Zarr format version to use (2 or 3). Defaults to 3.
        consolidated (bool, optional):
            Whether to write consolidated metadata. Defaults to False.
        engine (str, optional):
            Backend engine for loading radar files (e.g., "iris", "nexradlevel2"). Defaults to "nexradlevel2".
        batch_size (int, optional):
            Number of files per Dask batch. Defaults to the number of available CPU cores.
        **kwargs:
            Additional keyword arguments passed to the Zarr writer.

    Returns:
        None

    Notes:
        - Dask is used to parallelize file loading, but writing is done sequentially
          to preserve dataset consistency.
        - A local Dask cluster is spun up with a dashboard at http://127.0.0.1:8785.
        - Batch size is auto-determined if not specified.
    """
    import gc
    from functools import partial

    from dask import bag as db
    from dask.distributed import Client, LocalCluster

    from .dtree_radar import (
        radar_datatree,  # Safe local import to avoid circular issues
    )

    cluster = LocalCluster(dashboard_address="127.0.0.1:8785", memory_limit="10GB")
    client = Client(cluster)

    builder = partial(radar_datatree, engine=engine)

    if not batch_size:
        batch_size = sum(client.ncores().values())

    for radar_files_batch in batch(radar_files, n=batch_size):
        bag = db.from_sequence(radar_files_batch, npartitions=batch_size).map(builder)
        dtree_list: list[DataTree] = bag.compute()

        for dtree in dtree_list:
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
                    **kwargs,
                )

        del bag, dtree_list
        gc.collect()


def append_files(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    zarr_store: str,
    process_mode: Literal["sequential", "parallel"] = "sequential",
    engine: str = "iris",
    **kwargs,
) -> None:
    """
    Append radar files to a Zarr store using either sequential or parallel processing.

    This function serves as a unified interface for appending radar data into a Zarr store.
    It supports both serial and Dask-parallel strategies, controlled via the `mode` argument.
    Internally, it delegates to `append_sequential` or `append_parallel`.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            A list or generator of radar file paths to be appended.
        append_dim (str):
            The dimension name to append data along (e.g., "vcp_time").
        zarr_store (str):
            Path to the destination Zarr store on disk or cloud.
        process_mode (Literal["sequential", "parallel"], optional):
            Whether to use sequential or parallel processing. Defaults to "sequential".
        engine (str, optional):
            Backend engine used for reading radar data. Defaults to "iris".
            Options: "iris", "nexradlevel2", "odim".
        **kwargs:
            Extra keyword arguments passed to `append_sequential` or `append_parallel`.
            Common examples include:
              - zarr_format (int): Zarr format version (2 or 3).
              - consolidated (bool): Enable consolidated metadata.

    Raises:
        ValueError:
            If an unsupported mode is provided.

    Example:
        >>> append_files(["file1.RAW", "file2.RAW"], append_dim="vcp_time", zarr_store="output.zarr")
        >>> append_files(files, append_dim="vcp_time", zarr_store="s3://bucket/zarr", process_mode="parallel")
    """
    if process_mode == "sequential":
        kwargs.pop("batch_size", None)
        append_sequential(
            radar_files,
            append_dim=append_dim,
            zarr_store=zarr_store,
            engine=engine,
            **kwargs,
        )
    elif process_mode == "parallel":
        append_parallel(
            radar_files,
            append_dim=append_dim,
            zarr_store=zarr_store,
            engine=engine,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported mode: {process_mode}. Use 'sequential' or 'parallel'."
        )
