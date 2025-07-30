from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Literal

import icechunk
from dask.distributed import LocalCluster

from .executor import append_parallel, append_sequential


def convert_files(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    repo: icechunk.Repository,
    cluster: LocalCluster | object | None = None,
    process_mode: Literal["sequential", "parallel"] = "sequential",
    engine: str = "iris",
    remove_strings: bool = True,
    log_file: str = None,
    **kwargs,
) -> None:
    """
    Append radar files to a Zarr store using either sequential or parallel processing.

    This function serves as a unified interface for appending radar data into a Zarr store.
    It supports both serial and Dask-parallel strategies, controlled via the `process_mode` argument.
    Internally, it delegates to `append_sequential` or `append_parallel`.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            A list or generator of radar file paths to be appended.
        append_dim (str):
            The dimension name to append data along (e.g., "vcp_time").
        repo (icechunk.Repository):
            Icechunk repository object managing the Zarr store.
        cluster (optional):
            Dask cluster for distributed processing. Required for parallel mode, ignored for sequential mode.
        process_mode (Literal["sequential", "parallel"], optional):
            Whether to use sequential or parallel processing. Defaults to "sequential".
        engine (str, optional):
            Backend engine used for reading radar data. Defaults to "iris".
            Options: "iris", "nexradlevel2", "odim".
        remove_strings (bool, optional):
            Whether to remove variables of string dtype from the dataset before writing.
            This is necessary because Zarr v3 currently lacks full support for string dtypes.
            See: https://github.com/zarr-developers/zarr-python/pull/2874
            This option will be removed once native string support is available in Zarr v3.
        log_file (str, optional):
            Path to log file for problematic files. If None, uses "output.txt" in current directory.
        **kwargs:
            Additional keyword arguments passed to the underlying writer functions.
            Common examples include:
              - zarr_format (int): Zarr format version (2 or 3).
              - consolidated (bool): Enable consolidated metadata.

    Raises:
        ValueError:
            If an unsupported process_mode is provided.

    Example:
        >>> convert_files(["file1.RAW", "file2.RAW"], append_dim="vcp_time", repo=icechunk.repository)
        >>> convert_files(files, append_dim="vcp_time", repo=icechunk.Repository, process_mode="parallel")
    """
    if process_mode == "sequential":
        kwargs.pop("batch_size", None)
        append_sequential(
            radar_files,
            append_dim=append_dim,
            repo=repo,
            engine=engine,
            remove_strings=remove_strings,
            **kwargs,
        )
    elif process_mode == "parallel":
        append_parallel(
            radar_files,
            append_dim=append_dim,
            repo=repo,
            engine=engine,
            remove_strings=remove_strings,
            cluster=cluster,
            log_file=log_file,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported mode: {process_mode}. Use 'sequential' or 'parallel'."
        )
