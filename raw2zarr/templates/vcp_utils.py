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
            print(
                f"🔍 Detected active Dask client with {len(client.scheduler_info()['workers'])} workers"
            )
        except (ImportError, ValueError):
            # No active Dask client found
            use_parallel = False
            print("🔍 No active Dask client found, using sequential template creation")

    if use_parallel and len(vcp_time_mapping) > 1:
        # Parallel VCP template creation with shared template manager
        from dask.distributed import get_client

        print(
            f"🚀 Creating {len(vcp_time_mapping)} VCP templates in parallel using Client.map()..."
        )

        # Load config data locally and scatter to workers (avoids file path issues on remote workers)
        client = get_client()
        local_template_manager = VcpTemplateManager()

        # Pre-load config data locally
        scan_config_data = local_template_manager.config
        vcp_config_data = local_template_manager.vcp_config

        # Scatter config data to workers
        scan_config_future = client.scatter(scan_config_data, broadcast=True)
        vcp_config_future = client.scatter(vcp_config_data, broadcast=True)

        def create_vcp_template_optimized(vcp_data):
            """Optimized VCP template creation with pre-loaded config data"""
            (
                vcp_name,
                vcp_info,
                base_radar_info,
                append_dim,
                remove_strings,
                scan_config,
                vcp_config,
            ) = vcp_data

            # Create template manager with pre-loaded data (no file I/O on remote workers)
            template_mgr = VcpTemplateManager()
            template_mgr._full_config = scan_config  # Inject pre-loaded config
            template_mgr._vcp_configs = vcp_config  # Inject pre-loaded VCP config

            vcp_radar_info = base_radar_info.copy()
            vcp_radar_info["vcp"] = vcp_name

            vcp_tree = template_mgr.create_empty_vcp_tree(
                radar_info=vcp_radar_info,
                append_dim=append_dim,
                remove_strings=remove_strings,  # Keep per-VCP string removal as you noted
                append_dim_time=vcp_info[
                    "timestamps"
                ],  # Must use all timestamps for proper indexing
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
                scan_config_future,
                vcp_config_future,
            )
            for vcp_name, vcp_info in vcp_time_mapping.items()
        ]

        # Use Client.map for fastest execution
        futures = client.map(create_vcp_template_optimized, vcp_data_list)
        results = client.gather(futures)

        # Convert results to dictionary
        vcp_trees = {vcp_name: vcp_tree for vcp_name, vcp_tree in results}
        print(f"✅ Completed {len(vcp_trees)} VCP templates in parallel")

    else:
        # Sequential VCP template creation (fallback or single VCP)
        print(f"🔄 Creating {len(vcp_time_mapping)} VCP templates sequentially...")
        template_manager = VcpTemplateManager()
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
            print(f"  ✅ Created template for {vcp_name}")

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

    return vcp_time_mapping
