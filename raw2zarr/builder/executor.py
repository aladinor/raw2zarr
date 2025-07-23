from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

import icechunk

from ..templates.template_utils import remove_string_vars
from ..templates.vcp_utils import create_vcp_time_mapping
from ..transform.encoding import dtree_encoding
from ..writer.writer_utils import (
    init_zarr_store,
    resolve_zarr_write_options,
)
from ..writer.zarr_writer import dtree_to_zarr, write_dtree_region
from .builder_utils import extract_file_metadata
from .dtree_radar import radar_datatree

warnings.filterwarnings(
    "ignore",
    message=".*The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
    module="zarr.codecs.vlen_utf8",
)

# Suppress warning about `StringDType()` dtype
warnings.filterwarnings(
    "ignore",
    message=r".*The dtype `StringDType\(\)` is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
    module=r"zarr\.core\.array",
)


def append_sequential(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    repo: icechunk.Repository,
    zarr_format: int = 3,
    engine: str = "iris",
    mode: str = "a",
    remove_strings: bool = True,
    branch: str = "main",
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
        session (str):
            Icechunk session object.
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
        session = repo.writable_session(branch=branch)
        try:
            dtree = radar_datatree(file, engine=engine)

            if not dtree:
                print(f"[Warning] empty DataTree {file}")
                continue

            # TODO: remove this after strings are supported by zarr v3
            if remove_strings:
                dtree = remove_string_vars(dtree)
                dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)

            group_path = dtree.groups[1]

            writer_args = resolve_zarr_write_options(
                store=session.store,
                group_path=group_path,
                encoding=dtree.encoding,
                default_mode=mode,
                append_dim=append_dim,
                zarr_format=zarr_format,
            )
            dtree_to_zarr(dtree, **writer_args)

            snapshot_id = session.commit(f"Added file {file}")
            print(f"[icechunk] Committed {file} as snapshot {snapshot_id}")
        except Exception as e:
            print(f"[Warning] Failed to process {file}: {e}")
            continue


def append_parallel(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    repo: icechunk.Repository,
    zarr_format: int = 3,
    consolidated: bool = False,
    engine: str = "nexradlevel2",
    dashboard_address: str = "127.0.0.1:8785",
    branch: str = "main",
    remove_strings: bool = True,
) -> None:
    import logging

    import dask
    from dask.distributed import Client, LocalCluster
    from icechunk.distributed import merge_sessions

    logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)

    session = repo.writable_session(branch=branch)

    cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit="10GB")
    client = Client(
        cluster,
        timeout="60s",
        heartbeat_interval="10s",
    )

    metadata_tasks = [
        dask.delayed(extract_file_metadata)(f, engine) for f in radar_files
    ]
    metadata_results = dask.compute(*metadata_tasks)
    append_dim_time, vcps = zip(*metadata_results)
    append_dim_time = list(append_dim_time)
    vcps = list(vcps)
    file_indices = list(enumerate(radar_files))

    vcp_time_mapping = create_vcp_time_mapping(append_dim_time, vcps, file_indices)

    remaining_files = init_zarr_store(
        files=file_indices,
        session=session,
        append_dim=append_dim,
        engine=engine,
        zarr_format=zarr_format,
        consolidated=consolidated,
        vcp_time_mapping=vcp_time_mapping,
    )

    session = repo.writable_session(branch=branch)
    # TODO: add if statement to check if use region to fill empty store or
    with session.allow_pickling():
        file_vcp_mapping = {}

        # Always use VCP-specific mapping since we now use the refactored approach for all scenarios
        for vcp_name, vcp_info in vcp_time_mapping.items():
            for local_idx, file_info in enumerate(vcp_info["files"]):
                # Map files to their VCP-specific time indices
                file_vcp_mapping[file_info["filepath"]] = {
                    "time_index": local_idx,  # VCP-specific time index
                    "vcp": vcp_name,
                }

        tasks = [
            dask.delayed(write_dtree_region)(
                f,
                file_vcp_mapping.get(f, {"time_index": i, "vcp": None})["time_index"],
                session,
                append_dim,
                engine,
                zarr_format,
                consolidated,
                remove_strings,
            )
            for i, f in remaining_files
        ]
        sessions = dask.compute(*tasks, scheduler=client)

    session = merge_sessions(session, *sessions)
    session.commit("write Nexrad files to zarr store")
