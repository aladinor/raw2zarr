from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

import icechunk
from dask.distributed import LocalCluster

from ..templates.template_ops import (
    create_vcp_template_in_memory,
    merge_data_into_template,
)
from ..templates.template_utils import remove_string_vars
from ..transform.encoding import dtree_encoding
from ..writer.writer_utils import (
    check_cords,
    init_zarr_store,
    resolve_zarr_write_options,
)
from ..writer.zarr_writer import dtree_to_zarr, write_dtree_region
from .builder_utils import _log_problematic_file

# _log_problematic_file not available on Coiled workers - using local logging
from .dtree_radar import radar_datatree
from .metadata_processor import process_metadata_and_create_vcp_mapping

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
    vcp_config_file: str = "vcp_nexrad.json",
    **kwargs,
) -> None:
    """
    Append radar files to a Zarr store sequentially.

    This function loads each radar file one at a time, converts it to a DataTree,
    and appends it to the specified Zarr store. It is suitable for smaller datasets
    or environments where parallel processing is not needed.

    Supports dynamic NEXRAD scans (SAILS, MRLE, AVSET) by detecting temporal slices
    within each file and writing each slice as a separate time step.

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
        vcp_config_file (str, optional):
            VCP configuration file name for NEXRAD data. Defaults to "vcp_nexrad.json".
        **kwargs:
            Additional options passed to the Zarr writer.

    Returns:
        None

    Note:
        For dynamic scans (SAILS, MRLE), a single file may contain multiple temporal
        slices. Each slice is written as a separate time step in the Zarr store.
    """
    for file_idx, file in enumerate(radar_files):
        # Extract metadata for this file
        if engine == "nexradlevel2":
            from .builder_utils import extract_single_metadata

            metadata_entries = extract_single_metadata((file_idx, file), engine)
            # Returns list: 1 entry for STANDARD, 4 entries for MESO-SAILS×3
        else:
            # IRIS/ODIM: single entry, no dynamic scans
            metadata_entries = [(file_idx, file, None, None, 0, None, "STANDARD", None)]

        # Loop through temporal slices (may be multiple per file)
        for entry in metadata_entries:
            (
                _,
                filepath,
                timestamp,
                vcp,
                slice_id,
                sweep_indices,
                scan_type,
                elevation_angles,
            ) = entry

            # Skip corrupted files
            if scan_type == "CORRUPTED":
                print(f"[Warning] Skipping corrupted file: {filepath}")
                continue

            # Determine if dynamic
            is_dynamic = scan_type not in ["STANDARD"]

            session = repo.writable_session(branch=branch)
            try:
                # Step 1: Create in-memory template for NEXRAD files with valid VCP
                # This ensures all sweeps exist (including NaN-filled missing ones)
                template = None
                if engine == "nexradlevel2" and vcp:
                    try:
                        template = create_vcp_template_in_memory(
                            vcp=vcp,
                            append_dim=append_dim,
                            vcp_config_file=vcp_config_file,
                        )
                    except Exception as e:
                        print(
                            f"[Warning] Failed to create template for {vcp}: {e}. Proceeding without template."
                        )

                # Step 2: Load actual data from file
                dtree = radar_datatree(
                    filepath,
                    engine=engine,
                    append_dim=append_dim,
                    is_dynamic=is_dynamic,
                    sweep_indices=sweep_indices,
                    elevation_angles=elevation_angles,
                    vcp_config_file=vcp_config_file,
                )

                if not dtree:
                    print(f"[Warning] empty DataTree {filepath}")
                    continue

                # Step 3: Merge actual data into template (if template exists)
                # This fills in actual values while keeping missing sweeps as NaN
                if template is not None:
                    dtree = merge_data_into_template(template, dtree)

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

                snapshot_id = session.commit(
                    f"Added {scan_type} slice {slice_id} from {file}"
                )
                print(
                    f"[icechunk] Committed {scan_type} slice {slice_id} from {file} as snapshot {snapshot_id}"
                )
            except Exception as e:
                print(f"[Warning] Failed to process {filepath} slice {slice_id}: {e}")
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

    # Process metadata and create VCP time mapping
    result = process_metadata_and_create_vcp_mapping(
        client=client,
        radar_files=radar_files,
        engine=engine,
        skip_vcps=skip_vcps,
        log_file=log_file,
        generate_samples=generate_samples,
        sample_percentage=sample_percentage,
        samples_output_path=samples_output_path,
    )

    if result is None:
        return  # No valid files found

    vcp_time_mapping = result.vcp_time_mapping
    valid_files = result.valid_files
    import time

    start = time.time()
    remaining_files = init_zarr_store(
        files=valid_files,
        session=session,
        append_dim=append_dim,
        engine=engine,
        zarr_format=zarr_format,
        consolidated=consolidated,
        vcp_time_mapping=vcp_time_mapping,
        vcp_config_file=vcp_config_file,
        remove_strings=remove_strings,
    )
    elapsed = time.time() - start
    print(f"Time to initialize template in {elapsed:.4f}s")
    session = repo.writable_session(branch=branch)
    fork = session.fork()

    file_vcp_mapping = {}
    for vcp_name, vcp_info in vcp_time_mapping.items():
        for local_idx, file_info in enumerate(vcp_info["files"]):
            file_index = file_info["file_index"]
            scan_type = file_info.get("scan_type", "STANDARD")

            is_dynamic = scan_type not in ["STANDARD"]

            file_vcp_mapping[file_index] = {
                "time_index": local_idx,
                "vcp": vcp_name,
                "filepath": file_info["filepath"],
                "is_dynamic": is_dynamic,
                "sweep_indices": file_info.get("sweep_indices"),
                "scan_type": scan_type,
                "elevation_angles": file_info.get("elevation_angles"),
            }

    def write_single_file(file_info):
        """Write a single file using region writing - optimized for Client.map()"""
        file_idx, input_file = file_info
        try:
            meta = file_vcp_mapping.get(
                file_idx,
                {
                    "time_index": file_idx,
                    "vcp": None,
                    "is_dynamic": False,
                    "sweep_indices": None,
                    "scan_type": None,
                    "elevation_angles": None,
                },
            )
            return write_dtree_region(
                input_file,
                meta["time_index"],
                fork,
                append_dim,
                engine,
                zarr_format,
                consolidated,
                remove_strings,
                is_dynamic=meta.get("is_dynamic", False),
                sweep_indices=meta.get("sweep_indices"),
                elevation_angles=meta.get("elevation_angles"),
                vcp_config_file=vcp_config_file,
            )
        except Exception as e:
            return {
                "error": f"Write operation failed: {str(e)}",
                "file": input_file,
                "index": file_idx,
            }

    start = time.time()
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
    print(write_failed_files)
    # Log all write failures locally
    for file, error_msg in write_failed_files:
        _log_problematic_file(file, error_msg, log_file)

    # Only merge successful sessions
    if successful_sessions:
        session.merge(*successful_sessions)
        session.commit(
            f"writing {len(successful_sessions)}/{len(radar_files)} radar files to zarr store"
        )
        print(
            f"Wrote {len(radar_files)} radar files to zarr store, with {len(successful_sessions)} time steps"
        )
    else:
        print("No files wrote")
    elapsed = time.time() - start
    print(f"Time to write files in {elapsed:.4f}s")
    check_cords(repo)
