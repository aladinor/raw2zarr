from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

import icechunk
from dask.distributed import LocalCluster

from ..templates.template_utils import remove_string_vars
from ..templates.vcp_utils import create_vcp_time_mapping
from ..transform.encoding import dtree_encoding
from ..writer.writer_utils import (
    init_zarr_store,
    resolve_zarr_write_options,
)
from ..writer.zarr_writer import dtree_to_zarr, write_dtree_region
from .builder_utils import (
    _log_problematic_file,
    extract_single_metadata,
    generate_vcp_samples,
)

# _log_problematic_file not available on Coiled workers - using local logging
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
        repo (icechunk.Repository):
            Icechunk repository object for managing the Zarr store.
        zarr_format (int, optional):
            Zarr format version to use (2 or 3). Defaults to 3.
        engine (str, optional):
            Radar file reading engine (e.g., "iris", "nexradlevel2"). Defaults to "iris".
        mode (str, optional):
            Zarr write mode. Defaults to "a".
        remove_strings (bool, optional):
            Whether to remove variables of string dtype. Defaults to True.
        branch (str, optional):
            Git-like branch name for icechunk versioning. Defaults to "main".
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
    branch: str = "main",
    remove_strings: bool = True,
    cluster: LocalCluster | object | None = None,
    skip_vcps: list = None,
    log_file: str = None,
    generate_samples: bool = False,
    sample_percentage: float = 15.0,
    samples_output_path: str = None,
    vcp_config_file: str = "vcp_nexrad.json",
) -> None:
    """
    Append radar files to a Zarr store in parallel using Dask.

    This function processes multiple radar files concurrently using Dask distributed
    computing. It creates VCP-specific templates and uses region writing for efficient
    parallel data ingestion. Requires icechunk 1.0+ with Session.fork() API.

    Parameters:
        radar_files (Iterable[str | os.PathLike]):
            Paths to the radar data files to be appended.
        append_dim (str):
            Name of the dimension along which data should be appended (e.g., "vcp_time").
        repo (icechunk.Repository):
            Icechunk repository object for managing the Zarr store.
        zarr_format (int, optional):
            Zarr format version to use (2 or 3). Defaults to 3.
        consolidated (bool, optional):
            Whether to consolidate Zarr metadata. Defaults to False.
        engine (str, optional):
            Radar file reading engine (e.g., "iris", "nexradlevel2"). Defaults to "nexradlevel2".
        branch (str, optional):
            Git-like branch name for icechunk versioning. Defaults to "main".
        remove_strings (bool, optional):
            Whether to remove variables of string dtype. Defaults to True.
        cluster:
            Pre-configured Dask cluster (e.g., LocalCluster, Coiled cluster). Required for parallel processing.
        skip_vcps (list, optional):
            List of VCP patterns to skip during processing (e.g., ["VCP-31", "VCP-32"]). Defaults to None.
        log_file (str, optional):
            Path to log file for problematic files. If None, uses "output.txt" in current directory.
        generate_samples (bool, optional):
            Whether to generate VCP validation samples. Defaults to False.
        sample_percentage (float, optional):
            Percentage of files to sample for validation when generate_samples=True. Defaults to 15.0.
        samples_output_path (str, optional):
            Path to save VCP samples JSON file. If None, doesn't save to file.

    Returns:
        None

    Note:
        This function uses the new icechunk 1.0+ Session.fork() API for parallel
        processing. The old allow_pickling() context manager is no longer supported.
    """
    import logging

    from dask.distributed import Client

    logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)

    if not cluster:
        raise ValueError(
            "Cluster is required for parallel processing. Please provide a Dask cluster."
        )

    client = Client(cluster, timeout="60s", heartbeat_interval="10s")

    session = repo.writable_session(branch=branch)

    # Ultra-fast graph construction with Client.map()
    radar_files_with_indices = list(enumerate(radar_files))
    futures = client.map(
        extract_single_metadata, radar_files_with_indices, engine=engine
    )
    metadata_results = client.gather(futures)

    # Filter out problematic files and unwanted VCPs
    valid_results = []
    valid_files = []
    problematic_files = []
    skipped_vcps = []

    for original_index, file, result in metadata_results:
        if result[0] != "ERROR":
            timestamp, vcp_number = result
            vcp_name = f"VCP-{vcp_number}"

            # Check if this VCP should be skipped
            if skip_vcps and vcp_name in skip_vcps:
                skipped_vcps.append((file, vcp_name))
                continue

            # Valid result with (timestamp, vcp_number)
            valid_results.append(result)
            valid_files.append((original_index, file))
        else:
            # Problematic file with error info
            problematic_files.append((file, result[1]))

    # Log all problematic files locally (metadata extraction failures)
    for file, error_msg in problematic_files:
        _log_problematic_file(file, error_msg, log_file)

    # Log skipped VCPs locally
    for file, vcp_name in skipped_vcps:
        _log_problematic_file(
            file, f"Skipped {vcp_name} (configured to skip)", log_file
        )

    if not valid_results:
        print("❌ No valid files found after filtering problematic files.")
        return

    # Report file filtering statistics
    total_skipped = len(radar_files) - len(valid_results)
    if total_skipped > 0:
        if skipped_vcps:
            print(f"Skipped {len(skipped_vcps)} files from filtered VCPs: {skip_vcps}")
        problematic_count = len(problematic_files)
        if problematic_count > 0:
            print(
                f"Filtered out {problematic_count} problematic files (see output.txt for details)"
            )
        print(f"Processing {len(valid_results)} valid files")

    append_dim_time, vcps = zip(*valid_results)
    file_indices = valid_files
    append_dim_time = list(append_dim_time)
    vcps = list(vcps)

    vcp_time_mapping = create_vcp_time_mapping(append_dim_time, vcps, file_indices)

    vcp_names = list(vcp_time_mapping.keys())
    total_files = sum(info["file_count"] for info in vcp_time_mapping.values())
    print(f"Discovered {len(vcp_names)} VCP patterns in {total_files} files:")
    print(f"VCPs found: {', '.join(vcp_names)}")

    # Generate VCP validation samples if requested
    if generate_samples:
        generate_vcp_samples(
            vcp_time_mapping=vcp_time_mapping,
            sample_percentage=sample_percentage,
            output_path=samples_output_path,
        )

    remaining_files = init_zarr_store(
        files=file_indices,
        session=session,
        append_dim=append_dim,
        engine=engine,
        zarr_format=zarr_format,
        consolidated=consolidated,
        vcp_time_mapping=vcp_time_mapping,
        vcp_config_file=vcp_config_file,
    )
    session = repo.writable_session(branch=branch)
    fork = session.fork()

    file_vcp_mapping = {}
    for vcp_name, vcp_info in vcp_time_mapping.items():
        for local_idx, file_info in enumerate(vcp_info["files"]):
            file_vcp_mapping[file_info["filepath"]] = {
                "time_index": local_idx,
                "vcp": vcp_name,
            }

    def write_single_file(file_info):
        """Write a single file using region writing - optimized for Client.map()"""
        i, input_file = file_info
        try:
            meta = file_vcp_mapping.get(input_file, {"time_index": i, "vcp": None})
            return write_dtree_region(
                input_file,
                meta["time_index"],
                fork,
                append_dim,
                engine,
                zarr_format,
                consolidated,
                remove_strings,
            )
        except Exception as e:
            # Don't log from remote workers - return error info for local logging
            return {"error": f"Write operation failed: {str(e)}", "file": input_file}

    print(
        f"⚡ Using Client.map() for fastest graph construction with {len(remaining_files)} files"
    )

    # Ultra-fast graph construction with Client.map()
    write_futures = client.map(write_single_file, remaining_files)
    write_results = client.gather(write_futures)

    # Separate successful sessions from failed files
    successful_sessions = []
    write_failed_files = []

    for result in write_results:
        if isinstance(result, dict) and "error" in result:
            # Failed file
            write_failed_files.append((result["file"], result["error"]))
        else:
            # Successful session
            successful_sessions.append(result)

    # Log all write failures locally
    for file, error_msg in write_failed_files:
        _log_problematic_file(file, error_msg, log_file)

    # Only merge successful sessions
    if successful_sessions:
        session.merge(*successful_sessions)
        session.commit(
            f"writing {len(successful_sessions)}/{len(radar_files)} radar files to zarr store"
        )
        print(f"Wrote {len(radar_files)} radar files to zarr store")
