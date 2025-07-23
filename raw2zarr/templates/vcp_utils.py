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
):
    """
    Create individual xarray templates for each VCP and combine them into a single DataTree.

    This approach ensures proper dimension_names metadata that xarray needs for reading.

    Args:
        vcp_time_mapping: VCP mapping with timestamps and file info
        base_radar_info: Base radar metadata dict
        append_dim: Dimension name for appending (e.g., 'vcp_time')
        remove_strings: Whether to remove string variables

    Returns:
        xarray.DataTree: Combined multi-VCP template with proper metadata
    """
    from ..transform.encoding import dtree_encoding
    from .template_manager import VcpTemplateManager

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
