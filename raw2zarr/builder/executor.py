from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

import icechunk
import pandas as pd
from icechunk.session import Session
from xarray import DataTree

from ..templates.template_utils import remove_string_vars
from ..transform.encoding import dtree_encoding
from ..utils import batch
from ..writer.writer_utils import (
    drop_vars_region,
    resolve_zarr_write_options,
    zarr_store_has_append_dim,
)
from ..writer.zarr_writer import dtree_to_zarr
from .builder_utils import extract_file_metadata
from .dtree_radar import radar_datatree

# Suppress the specific warning about `vlen-utf8` codec from Zarr
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
    engine: str = "nexradlevel2",
    batch_size: int = None,
    branch: str = "main",
    mode: str = "a",
    remove_strings: bool = True,
    **kwargs,
) -> None:
    """
    Append radar files to a Zarr store in parallel using Dask.

    Parameters:
        radar_files: List of input radar files.
        append_dim: Dimension to append along (e.g., "vcp_time").
        zarr_store: Output path.
        zarr_format: Zarr version (default: 3).
        consolidated: Whether to write consolidated metadata.
        engine: Radar parsing engine (e.g., "iris", "nexradlevel2").
        batch_size: How many files to process in parallel.
        **kwargs: Passed to the writer.
    """
    import gc
    from functools import partial

    from dask import bag as db
    from dask.distributed import Client, LocalCluster

    from .dtree_radar import radar_datatree

    cluster = LocalCluster(dashboard_address="127.0.0.1:8785", memory_limit="10GB")
    client = Client(cluster)

    builder = partial(radar_datatree, engine=engine)

    if not batch_size:
        batch_size = sum(client.ncores().values()) - 2
    try:
        for radar_files_batch in batch(radar_files, n=batch_size):
            bag = db.from_sequence(radar_files_batch, npartitions=batch_size).map(
                builder
            )
            dtree_list: list[DataTree] = bag.compute()

            for idx, dtree in enumerate(dtree_list):
                if not dtree:
                    continue
                # TODO: remove this after strings are supported by zarr v3
                if remove_strings:
                    dtree = remove_string_vars(dtree)
                    dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)

                session = repo.writable_session(branch)
                group_path = dtree.groups[1]
                writer_args = resolve_zarr_write_options(
                    store=session.store,
                    encoding=dtree.encoding,
                    group_path=group_path,
                    default_mode=mode,
                    append_dim=append_dim,
                    zarr_format=zarr_format,
                )

                dtree_to_zarr(dtree, **writer_args)
                snapshot_id = session.commit(f"Added file {radar_files_batch[idx]}")
                print(
                    f"[icechunk] Committed {radar_files_batch[idx]} as snapshot {snapshot_id}"
                )
            del bag, dtree_list
            gc.collect()
    finally:
        client.shutdown()
        client.close()
        cluster.close()


def append_parallel_region(
    radar_files: Iterable[str | os.PathLike],
    append_dim: str,
    repo: icechunk.Repository,
    zarr_format: int = 3,
    consolidated: bool = False,
    engine: str = "nexradlevel2",
    dashboard_address: str = "127.0.0.1:8785",
    branch: str = "main",
    remove_strings: bool = True,
    **kwargs,
) -> None:
    import dask
    from dask.distributed import Client, LocalCluster
    from icechunk.distributed import merge_sessions

    session = repo.writable_session(branch=branch)

    # Configure cluster to handle task key conflicts more gracefully
    cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit="10GB")
    client = Client(
        cluster,
        # Configure client to be more tolerant of task key reuse
        timeout="60s",
        heartbeat_interval="10s",
    )

    metadata_tasks = [
        dask.delayed(extract_file_metadata)(f, engine) for f in radar_files
    ]
    metadata_results = dask.compute(*metadata_tasks)

    # Unpack timestamps and VCP numbers
    append_dim_time, vcps = zip(*metadata_results)
    append_dim_time = list(append_dim_time)
    vcps = list(vcps)
    file_indices = list(enumerate(radar_files))

    # Create VCP-time mapping for multi-VCP support
    vcp_time_mapping = _create_vcp_time_mapping(
        radar_files, append_dim_time, vcps, file_indices
    )

    remaining_files = _init_zarr_store(
        files=file_indices,
        session=session,
        append_dim=append_dim,
        engine=engine,
        zarr_format=zarr_format,
        consolidated=consolidated,
        size_append_dim=len(radar_files),
        append_dim_time=append_dim_time,
        vcp_time_mapping=vcp_time_mapping,
        **kwargs,
    )

    session = repo.writable_session(branch=branch)
    # TODO: add if statement to check if use region to fill empty store or
    with session.allow_pickling():
        # Create VCP-aware file mapping for multi-VCP scenarios
        file_vcp_mapping = {}
        if vcp_time_mapping and len(vcp_time_mapping) > 1:
            # Multi-VCP: map files to their VCP-specific time indices
            for vcp_name, vcp_info in vcp_time_mapping.items():
                start_idx, end_idx = vcp_info["time_range"]
                for local_idx, file_info in enumerate(vcp_info["files"]):
                    global_time_idx = start_idx + local_idx
                    file_vcp_mapping[file_info["filepath"]] = {
                        "time_index": global_time_idx,
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
        print(f"Issuing {len(tasks)} tasks with VCP-aware time mapping.")
        sessions = dask.compute(*tasks, scheduler=client)

    session = merge_sessions(session, *sessions)
    session.commit("write Nexrad files to zarr store")


def _init_zarr_store(
    files: list[tuple[int, str]],
    session: Session,
    append_dim: str,
    engine: str,
    zarr_format: int,
    consolidated: bool,
    size_append_dim: int,
    remove_strings: bool = True,
    append_dim_time: pd.Timestamp | None = None,
    vcp_time_mapping: dict | None = None,
    **kwargs,
) -> list[tuple[int, str]]:
    from ..templates.template_manager import VcpTemplateManager

    exis_zarr_store = zarr_store_has_append_dim(
        session.store,
        append_dim=append_dim,
    )
    if not exis_zarr_store:
        idx, first_file = files.pop(0)
        dtree = radar_datatree(first_file, engine=engine)
        vcp = dtree[dtree.groups[1]].attrs["scan_name"]
        radar_info = {
            "lon": dtree[vcp].longitude.item(),
            "lat": dtree[vcp].latitude.item(),
            "alt": dtree[vcp].altitude.item(),
            "crs_wkt": dtree[f"{vcp}/sweep_0"].ds["crs_wkt"].attrs,
            "reference_time": dtree[vcp].time_coverage_start.item(),
            "vcp": vcp,
            "instrument_name": dtree[vcp].attrs["instrument_name"],
            "volume_number": dtree[vcp].volume_number.item(),
            "platform_type": dtree[vcp].platform_type.item(),
            "instrument_type": dtree[vcp].instrument_type.item(),
            "time_coverage_start": dtree[vcp].time_coverage_start.item(),
            "time_coverage_end": dtree[vcp].time_coverage_end.item(),
        }

        template_manager = VcpTemplateManager()

        # Choose template based on VCP diversity
        if vcp_time_mapping and len(vcp_time_mapping) > 1:
            # Multi-VCP scenario: create template for all VCPs
            print(
                f"[DEBUG] Creating multi-VCP template for {len(vcp_time_mapping)} VCPs"
            )
            empty_tree = template_manager.create_multi_vcp_tree(
                radar_info=radar_info,
                vcp_time_mapping=vcp_time_mapping,
                append_dim=append_dim,
                remove_strings=remove_strings,
                dim_chunksize=kwargs.get("dim_chunksize"),
            )
        else:
            # Single VCP scenario: use original approach
            print(f"[DEBUG] Creating single-VCP template for VCP {vcp}")
            empty_tree = template_manager.create_empty_vcp_tree(
                radar_info=radar_info,
                append_dim=append_dim,
                size_append_dim=size_append_dim,
                remove_strings=remove_strings,
                append_dim_time=append_dim_time,
            )

        writer_args = resolve_zarr_write_options(
            store=session.store,
            group_path=None,
            encoding=empty_tree.encoding,
            append_dim=append_dim,
            zarr_format=zarr_format,
            consolidated=consolidated,
            compute=False,
        )

        dtree_to_zarr(empty_tree, **writer_args)
        # TODO: remove this after strings are supported by zarr v3
        if remove_strings:
            dtree = remove_string_vars(dtree)
            dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)
        writer_args = resolve_zarr_write_options(
            store=session.store,
            group_path=None,
            encoding=dtree.encoding,
            region={append_dim: slice(0, 1)},
            zarr_format=zarr_format,
            append_dim=append_dim,
        )
        dtree_append = drop_vars_region(dtree, append_dim=append_dim)
        dtree_to_zarr(dtree_append, **writer_args)
        session.commit("Initial commit: zarr store initialization")
        print("Initial commit: zarr store initialization")
        return files
    return files


def write_dtree_region(
    file: str,
    idx: int,
    session: icechunk.Session,
    append_dim: str,
    engine: str,
    zarr_format: int = 3,
    consolidated: bool = False,
    remove_strings=True,
    **kwargs,
) -> icechunk.Session:
    dtree = radar_datatree(file, engine=engine)
    # TODO: remove this after strings are supported by zarr v3
    if remove_strings:
        dtree = remove_string_vars(dtree)
        dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)

    region = {append_dim: slice(idx, idx + 1)}

    writer_args = dict(
        store=session.store,
        mode="a-",
        zarr_format=zarr_format,
        consolidated=consolidated,
        write_inherited_coords=True,
        region=region,
        **kwargs,
    )
    dtree_append = drop_vars_region(dtree, append_dim=append_dim)
    dtree_to_zarr(dtree_append, **writer_args)
    return session


def _create_vcp_time_mapping(
    radar_files: list[str],
    timestamps: list[pd.Timestamp],
    vcps: list[str],
    file_indices: list[tuple[int, str]],
) -> dict:
    """
    Create VCP-time mapping for multi-VCP template creation.

    Groups files by VCP and creates time blocks for each VCP pattern.

    Parameters:
        radar_files: List of radar file paths
        timestamps: List of timestamps extracted from files
        vcps: List of VCP numbers extracted from files
        file_indices: List of (index, filepath) tuples

    Returns:
        dict: VCP mapping structure with time blocks and file locations
    """
    from collections import defaultdict

    # Group files by VCP
    vcp_groups = defaultdict(list)

    for i, (file_idx, filepath) in enumerate(file_indices):
        vcp = f"VCP-{vcps[i]}"
        vcp_groups[vcp].append(
            {
                "file_index": file_idx,
                "filepath": filepath,
                "timestamp": timestamps[i],
                "original_position": i,
            }
        )

    # Sort each VCP group by timestamp
    for vcp in vcp_groups:
        vcp_groups[vcp].sort(key=lambda x: x["timestamp"])

    # Create time blocks for each VCP
    vcp_time_mapping = {}
    global_time_index = 0

    for vcp, files in vcp_groups.items():
        # Create continuous time block for this VCP
        start_time_idx = global_time_index
        end_time_idx = global_time_index + len(files)

        vcp_time_mapping[vcp] = {
            "files": files,
            "time_range": (start_time_idx, end_time_idx),
            "timestamps": [f["timestamp"] for f in files],
            "file_count": len(files),
        }

        global_time_index = end_time_idx

    # Debug output
    print(f"\n[DEBUG] Multi-VCP Analysis:")
    print(f"Total files: {len(radar_files)}")
    print(f"Unique VCPs found: {list(vcp_time_mapping.keys())}")

    for vcp, info in vcp_time_mapping.items():
        print(f"  {vcp}: {info['file_count']} files, time range {info['time_range']}")
        print(f"    Time span: {info['timestamps'][0]} to {info['timestamps'][-1]}")

    return vcp_time_mapping
