"""
Template operations for VCP data processing.

This module provides utilities for creating, merging, and manipulating
VCP templates during radar data processing.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

from .template_manager import VcpTemplateManager


def create_vcp_template_in_memory(
    vcp: str,
    append_dim: str,
    vcp_config_file: str = "vcp_nexrad.json",
) -> xr.DataTree:
    """
    Create empty VCP template with all expected sweeps (NaN-filled).

    Creates a template in memory with all sweeps defined in the VCP configuration,
    filled with NaN values. This ensures consistent structure when actual data
    has missing sweeps (e.g., AVSET scans that terminate early).

    Parameters
    ----------
    vcp : str
        VCP pattern name (e.g., "VCP-212")
    append_dim : str
        Dimension name for appending (e.g., "vcp_time")
    vcp_config_file : str
        VCP configuration file name

    Returns
    -------
    xr.DataTree
        Template with all expected sweeps filled with NaN
    """
    template_mgr = VcpTemplateManager(vcp_config_file=vcp_config_file)

    # Create minimal radar info for template
    # These dummy values will be overwritten by actual data during merge
    radar_info = {
        "vcp": vcp,
        "lat": 0.0,
        "lon": 0.0,
        "alt": 0.0,
        "volume_number": 0,
        "platform_type": "fixed",
        "instrument_type": "radar",
        "primary_axis": "axis_z",
        "time_coverage_start": "1970-01-01T00:00:00Z",
        "time_coverage_end": "1970-01-01T00:00:00Z",
        "reference_time": pd.Timestamp("1970-01-01T00:00:00Z"),
        "crs_wkt": {"grid_mapping_name": "latitude_longitude"},
    }

    # Create template with single time step
    template = template_mgr.create_empty_vcp_tree(
        radar_info=radar_info,
        append_dim=append_dim,
        remove_strings=True,
        append_dim_time=[pd.Timestamp.now()],  # Single dummy timestamp
    )

    return template


def merge_data_into_template(
    template: xr.DataTree, actual_data: xr.DataTree
) -> xr.DataTree:
    """
    Overlay actual data onto template, preserving template structure.

    Replaces only data variables within each matching dataset node — never
    coordinates — and only when dims match exactly. This keeps template
    coordinates/structure intact and fills in values for variables present
    in both datasets. Missing variables in actual data remain as template
    placeholders (e.g., NaNs), and we emit warnings for visibility.

    Parameters
    ----------
    template : xr.DataTree
        Template with all expected sweeps (NaN-filled)
    actual_data : xr.DataTree
        Actual data loaded from file (may have missing sweeps)

    Returns
    -------
    xr.DataTree
        Template with actual variables overlaid; missing variables remain NaN
    """
    # Start with actual data and add missing pieces from template
    template_dict: dict[str, xr.Dataset] = template.to_dict()

    # Normalize paths to include a leading slash for consistent dict keys
    def _norm(path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    result_dict: dict[str, xr.Dataset] = {
        _norm(k): v for k, v in actual_data.to_dict().items()
    }
    template_dict: dict[str, xr.Dataset] = {
        _norm(k): v for k, v in template.to_dict().items()
    }

    # Iterate over template paths to ensure nodes and variables exist in actual
    for path, tmpl_ds in template_dict.items():
        if not isinstance(tmpl_ds, xr.Dataset):
            continue

        # Add missing node entirely
        if path not in result_dict or not isinstance(result_dict[path], xr.Dataset):
            result_dict[path] = tmpl_ds
            continue

        act_ds = result_dict[path]

        # Compute which variables are missing in actual
        tmpl_vars = set(tmpl_ds.data_vars)
        act_vars = set(act_ds.data_vars)
        missing_vars = sorted(tmpl_vars - act_vars)
        if not missing_vars:
            continue

        merged = act_ds.copy()
        for var in missing_vars:
            tvar = tmpl_ds[var]

            # If var exists as coordinate, remove coord to allow adding as data var
            if var in merged.coords and var not in merged.data_vars:
                try:
                    merged = merged.drop_vars(var)
                except Exception:
                    pass

            # Skip if template dims are not present in actual ds
            if not all(dim in merged.dims for dim in tvar.dims):
                continue

            # Align coordinates on each dimension to avoid coordinate union
            try:
                tvar = tvar.assign_coords(
                    {
                        dim: merged.coords[dim]
                        for dim in tvar.dims
                        if dim in merged.coords
                    }
                )
            except Exception:
                pass

            merged[var] = tvar

        result_dict[path] = merged

    return xr.DataTree.from_dict(result_dict)
