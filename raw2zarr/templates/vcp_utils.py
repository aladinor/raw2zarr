"""Utilities for VCP (Volume Coverage Pattern) processing and template creation."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import xarray as xr


def create_multi_vcp_template(
    vcp_time_mapping: dict,
    base_radar_info: dict,
    append_dim: str,
    remove_strings: bool = True,
    use_parallel: bool = None,
    vcp_config_file: str = "vcp_nexrad.json",
):
    """
    Create individual xarray templates for each VCP and combine them into a single DataTree.

    This approach ensures proper dimension_names metadata that xarray needs for reading.
    When multiple VCPs are present and a Dask client is active, templates are created
    in parallel for improved performance.

    Args:
        vcp_time_mapping: VCP mapping with timestamps and file info
        base_radar_info: Base radar metadata dict
        append_dim: Dimension name for appending (e.g., 'vcp_time')
        remove_strings: Whether to remove string variables
        use_parallel: Whether to create VCP templates in parallel. If None, auto-detects
                      based on whether Dask client is available (default: None)
        vcp_config_file: VCP configuration file name in the config directory (default: "vcp_nexrad.json")

    Returns:
        xarray.DataTree: Combined multi-VCP template with proper metadata

    Performance:
        - Sequential: ~2-5 seconds per VCP (scales linearly)
        - Parallel: ~2-5 seconds total (scales with Dask workers)
        - Auto-detects parallel mode when process_mode="parallel" is used
    """
    from ..transform.encoding import dtree_encoding
    from .template_manager import VcpTemplateManager

    # Auto-detect parallel mode if not specified
    if use_parallel is None:
        try:
            from dask.distributed import get_client

            client = get_client()
            use_parallel = True

        except (ImportError, ValueError):
            # No active Dask client found
            use_parallel = False

    if use_parallel and len(vcp_time_mapping) > 1:
        # Parallel VCP template creation with shared template manager
        from dask.distributed import get_client

        # Load config data locally and scatter to workers (avoids file path issues on remote workers)
        client = get_client()
        local_template_manager = VcpTemplateManager(vcp_config_file=vcp_config_file)

        # Pre-load config data locally
        config_data = local_template_manager.config

        # Scatter config data to workers
        config_future = client.scatter(config_data, broadcast=True)

        def create_vcp_template_optimized(vcp_data):
            """Optimized VCP template creation with pre-loaded config data"""
            (
                vcp_name,
                vcp_info,
                radar_info_data,
                append_dim_data,
                remove_strings_data,
                config_data,
                vcp_config_file,
            ) = vcp_data

            # Create template manager with pre-loaded config (no file I/O on remote workers)
            template_mgr = VcpTemplateManager(vcp_config_file=vcp_config_file)
            template_mgr._unified_config = config_data  # Inject pre-loaded config

            radar_info_copy = radar_info_data.copy()
            radar_info_copy["vcp"] = vcp_name

            vcp_tree = template_mgr.create_empty_vcp_tree(
                radar_info=radar_info_copy,
                append_dim=append_dim_data,
                remove_strings=remove_strings_data,
                append_dim_time=vcp_info["timestamps"],
            )
            return vcp_name, vcp_tree[vcp_name]

        # Prepare data for parallel processing
        vcp_data_list = [
            (
                vcp_name,
                vcp_info,
                base_radar_info,
                append_dim,
                remove_strings,
                config_future,
                vcp_config_file,
            )
            for vcp_name, vcp_info in vcp_time_mapping.items()
        ]

        # Use Client.map for fastest execution
        futures = client.map(create_vcp_template_optimized, vcp_data_list)
        results = client.gather(futures)

        # Convert results to dictionary
        vcp_trees = {vcp_name: vcp_tree for vcp_name, vcp_tree in results}

    else:
        template_manager = VcpTemplateManager(vcp_config_file=vcp_config_file)
        vcp_trees = {}

        for vcp_name, vcp_info in vcp_time_mapping.items():
            vcp_radar_info = base_radar_info.copy()
            vcp_radar_info["vcp"] = vcp_name

            vcp_tree = template_manager.create_empty_vcp_tree(
                radar_info=vcp_radar_info,
                append_dim=append_dim,
                remove_strings=remove_strings,
                append_dim_time=vcp_info["timestamps"],  # VCP-specific timestamps
            )
            vcp_trees[vcp_name] = vcp_tree[vcp_name]
            print(f"Created template for {vcp_name}")

    final_tree = xr.DataTree.from_dict(vcp_trees)

    # Apply encoding to the combined tree to ensure all nodes have proper encoding
    final_tree.encoding = dtree_encoding(final_tree, append_dim=append_dim)
    return final_tree


def create_vcp_time_mapping(
    timestamps: list[pd.Timestamp],
    vcps: list[str],
    file_indices: list[tuple[int, str]],
) -> dict:
    """
    Create VCP-time mapping for multi-VCP template creation.

    Groups files by VCP and creates time blocks for each VCP pattern.

    Parameters:
        timestamps: List of timestamps extracted from files
        vcps: List of VCP numbers extracted from files
        file_indices: List of (index, filepath) tuples

    Returns:
        dict: VCP mapping structure with time blocks and file locations
    """
    # Group files by VCP
    vcp_groups = defaultdict(list)

    for i, (file_idx, filepath) in enumerate(file_indices):
        vcp = vcps[i]
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

    return vcp_time_mapping


def create_vcp_time_mapping_with_slices(
    metadata_results: list[tuple],
    valid_files: list[tuple],
) -> dict:
    """
    Create VCP-time mapping from flattened metadata results with temporal slices.

    This function handles both standard scans and dynamic scans with temporal slicing.
    For dynamic scans (SAILS, MRLE), multiple entries per file represent different
    temporal slices, which are treated as separate "virtual files" for template creation.

    Parameters
    ----------
    metadata_results : list of tuple
        Flattened metadata results where each entry is:
        (timestamp, vcp, slice_id, sweep_indices, scan_type, elevation_angles)
    valid_files : list of tuple
        List of (slice_index, filepath) tuples corresponding to metadata_results

    Returns
    -------
    dict
        VCP mapping structure with time blocks and file locations:
        {
            "VCP-212": {
                "files": [
                    {
                        "file_index": int,
                        "filepath": str,
                        "timestamp": pd.Timestamp,
                        "original_position": int,
                        "slice_id": int,
                        "sweep_indices": list,
                        "scan_type": str,
                        "elevation_angles": list | None,
                    },
                    ...
                ],
                "time_range": (start_idx, end_idx),
                "timestamps": [pd.Timestamp, ...],
                "file_count": int,
            },
            ...
        }
    """
    # Create filepath lookup from valid_files
    file_index_to_path = {idx: filepath for idx, filepath in valid_files}

    # Group by VCP
    vcp_groups = defaultdict(list)

    for i, (
        timestamp,
        vcp,
        slice_id,
        sweep_indices,
        scan_type,
        elevation_angles,
    ) in enumerate(metadata_results):
        vcp_groups[vcp].append(
            {
                "file_index": i,
                "filepath": file_index_to_path.get(i, ""),
                "timestamp": timestamp,
                "slice_id": slice_id,
                "sweep_indices": sweep_indices,
                "scan_type": scan_type,
                "elevation_angles": elevation_angles,
                "original_position": i,
            }
        )

    # Sort each VCP group by timestamp to ensure monotonic time
    for vcp in vcp_groups:
        vcp_groups[vcp].sort(key=lambda x: x["timestamp"])

    # Create time blocks for each VCP
    vcp_time_mapping = {}
    global_time_index = 0

    for vcp, entries in vcp_groups.items():
        # Create continuous time block for this VCP
        start_time_idx = global_time_index
        end_time_idx = global_time_index + len(entries)

        vcp_time_mapping[vcp] = {
            "files": entries,
            "time_range": (start_time_idx, end_time_idx),
            "timestamps": [entry["timestamp"] for entry in entries],
            "file_count": len(entries),
        }

        global_time_index = end_time_idx

    return vcp_time_mapping


def map_sweeps_to_vcp_indices(
    data_tree,
    vcp: str,
    sweep_indices: list[int] | None,
    elevation_angles: list[float] | None,
    vcp_config_file: str = "vcp_nexrad.json",
):
    """
    Map DataTree sweeps to VCP template indices using elevation angles and dimensions.

    For dynamic scans, this function maps file sweeps to their corresponding VCP
    template indices by matching elevation angles and range dimensions (to
    distinguish split-cut pairs).

    Parameters
    ----------
    data_tree : DataTree
        Loaded radar data with file sweep indices
    vcp : str
        VCP name (e.g., "VCP-212")
    sweep_indices : list[int] | None
        For SAILS/MRLE temporal slices: raw file sweep indices (e.g., [4,5,6,7,8,9])
        For AVSET/STANDARD: None (use sequential indices)
    elevation_angles : list[float] | None
        Elevation angles for this temporal slice (e.g., [0.48, 0.48, 1.32, 1.32])
    vcp_config_file : str
        VCP configuration file name

    Returns
    -------
    DataTree
        Tree with sweeps renumbered to match VCP template indices

    Example
    -------
    MESO-SAILS slice with file sweeps [4,5,6,7,8,9]:

    - Elevations: [0.48, 0.48, 1.32, 1.32, 1.8, 2.42]
    - VCP-212 config: [0.5, 0.5, 0.9, 0.9, 1.3, 1.3, 1.8, 2.4, ...]
    - Mapping: file sweep_4 → VCP sweep_0, file sweep_5 → VCP sweep_1, etc.
    - Returns tree with: sweep_0, sweep_1, sweep_4, sweep_5, sweep_6, sweep_7
    """
    from .template_manager import VcpTemplateManager

    # If no elevation angles provided, return as-is
    if elevation_angles is None:
        return data_tree

    # Load VCP config using template manager
    template_mgr = VcpTemplateManager(vcp_config_file=vcp_config_file)

    # Get VCP configuration
    try:
        vcp_info = template_mgr.config[vcp]
    except KeyError:
        # VCP not in config, return as-is
        return data_tree

    vcp_elevations = vcp_info["elevations"]
    vcp_range_dims = vcp_info["dims"]["range"]

    # Map file sweep indices to VCP template indices
    sweep_mapping = {}  # {file_sweep_idx: vcp_sweep_idx}

    # Use provided sweep_indices (SAILS/MRLE) or sequential indices (AVSET/STANDARD)
    indices_to_process = (
        sweep_indices
        if sweep_indices is not None
        else list(range(len(elevation_angles)))
    )

    for idx, file_sweep_idx in enumerate(indices_to_process):
        # Get elevation for this sweep
        elev = elevation_angles[idx]
        rounded_elev = round(elev, 1)

        # Get actual dimensions from the file sweep
        file_sweep_name = f"sweep_{file_sweep_idx}"
        if file_sweep_name not in data_tree.children:
            continue

        sweep_ds = data_tree[file_sweep_name].ds
        actual_range = len(sweep_ds.range) if "range" in sweep_ds.dims else 0

        # Find matching elevation + range in VCP config (for split-cut disambiguation)
        for vcp_idx, (vcp_elev, vcp_range) in enumerate(
            zip(vcp_elevations, vcp_range_dims)
        ):
            rounded_vcp_elev = round(vcp_elev, 1)
            elev_match = abs(rounded_elev - rounded_vcp_elev) < 0.15
            range_match = actual_range == vcp_range

            if elev_match and range_match:
                # Check if this VCP index hasn't been mapped yet
                if vcp_idx not in sweep_mapping.values():
                    sweep_mapping[file_sweep_idx] = vcp_idx
                    break

    # Build new tree with VCP sweep indices
    remapped_dict = {}

    # Copy root if it exists
    if hasattr(data_tree, "ds") and data_tree.ds is not None:
        remapped_dict["/"] = data_tree.ds

    # Copy sweeps with VCP indices
    for file_sweep_idx, vcp_idx in sweep_mapping.items():
        file_sweep_name = f"sweep_{file_sweep_idx}"
        vcp_sweep_name = f"sweep_{vcp_idx}"

        if file_sweep_name in data_tree.children:
            remapped_dict[vcp_sweep_name] = data_tree[file_sweep_name].ds

    return xr.DataTree.from_dict(remapped_dict)
