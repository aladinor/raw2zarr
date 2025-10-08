import warnings

import pandas as pd
from xarray import DataTree

from ..templates.template_manager import VcpTemplateManager


def slice_to_vcp_dimensions(
    dtree: DataTree,
    vcp: str,
    vcp_config_file: str = "vcp_nexrad.json",
) -> DataTree:
    """
    Slice data variables to match VCP config range dimensions.

    Ensures backward compatibility by trimming old data (with more range bins)
    to match current VCP config (with fewer range bins). This is necessary because
    VCP configurations evolved over time (e.g., VCP-212 from 2011 to 2025 reduced
    range bins in several sweeps).

    For parallel mode: Template is created from current VCP config, so old data
    must be sliced to match template dimensions before writing.

    Notes
    -----
    - Returns the original DataTree unchanged if no slicing is needed
    - All non-sweep children (metadata, etc.) are preserved
    - Dynamic scans (SAILS, MRLE) are not sliced and return unchanged
    - Does not modify the input DataTree (creates new one if slicing needed)

    Parameters
    ----------
    dtree : DataTree
        Loaded radar DataTree (will not be modified)
    vcp : str
        VCP pattern name (e.g., "VCP-212", "MESO-SAILS×2")
    vcp_config_file : str
        VCP configuration file name

    Returns
    -------
    DataTree
        Original DataTree if no slicing needed, or new DataTree with range
        dimensions sliced to match VCP config

    Warns
    -----
    UserWarning
        If VCP not found in config, config file missing, or other issues
    """
    # Skip if VCP is not provided
    if not vcp:
        return dtree

    # Skip dynamic scans - they don't have standard VCP numbers
    dynamic_patterns = ["SAILS", "MRLE", "AVSET", "MESO"]
    if any(pattern in vcp for pattern in dynamic_patterns) or "×" in vcp:
        return dtree

    # Validate VCP format
    if not vcp.startswith("VCP-"):
        warnings.warn(
            f"Unexpected VCP format: {vcp}. Expected 'VCP-XXX'. Skipping range slicing.",
            UserWarning,
        )
        return dtree

    # Load VCP configuration using VcpTemplateManager
    try:
        template_mgr = VcpTemplateManager(vcp_config_file=vcp_config_file)
        vcp_config = template_mgr.config
    except (FileNotFoundError, ValueError, PermissionError) as e:
        warnings.warn(
            f"Could not load VCP config file {vcp_config_file}: {e}. Skipping range slicing.",
            UserWarning,
        )
        return dtree

    # Get VCP-specific config
    if vcp not in vcp_config:
        warnings.warn(
            f"VCP {vcp} not found in config file. Skipping range slicing.", UserWarning
        )
        return dtree

    vcp_info = vcp_config[vcp]

    # Get expected range dimensions per sweep
    if "dims" not in vcp_info or "range" not in vcp_info["dims"]:
        return dtree

    expected_ranges = vcp_info["dims"]["range"]

    # Process sweeps and track if any slicing occurred
    processed_sweeps = {}
    any_sliced = False

    for child_name, child in dtree.children.items():
        # Keep non-sweep children as-is
        if not child_name.startswith("sweep_"):
            processed_sweeps[child_name] = child
            continue

        # Extract and validate sweep index
        try:
            sweep_idx = int(child_name.split("_")[1])
        except (IndexError, ValueError):
            # Invalid sweep name format - keep unchanged
            processed_sweeps[child_name] = child
            continue

        # Keep sweep unchanged if no config for this sweep index
        if sweep_idx >= len(expected_ranges):
            processed_sweeps[child_name] = child
            continue

        expected_range = expected_ranges[sweep_idx]

        # Validate expected range
        if not isinstance(expected_range, (int,)) or expected_range <= 0:
            # Invalid expected range - keep unchanged
            processed_sweeps[child_name] = child
            continue

        # Keep sweep unchanged if range dimension doesn't exist
        if "range" not in child.dims:
            processed_sweeps[child_name] = child
            continue

        actual_range = child.dims["range"]

        # Slice if actual > expected
        if actual_range > expected_range:
            sliced_child = child.isel(range=slice(None, expected_range))
            processed_sweeps[child_name] = sliced_child
            any_sliced = True
        else:
            processed_sweeps[child_name] = child

    # Return original if no slicing occurred
    if not any_sliced:
        return dtree

    # Rebuild DataTree with sliced sweeps using from_dict
    tree_dict = {}

    # Add root dataset if exists
    if dtree.ds is not None:
        tree_dict["/"] = dtree.ds

    # Add all children (sweeps and non-sweeps)
    for name, child in processed_sweeps.items():
        tree_dict[name] = child

    # Create new tree from dict
    new_dtree = DataTree.from_dict(tree_dict)

    return new_dtree


def exp_dim(dtree: DataTree, append_dim: str) -> DataTree:
    """
    Add a new dimension to all datasets in a DataTree and initialize it with a specific value.

    Parameters:
        dtree (DataTree): The DataTree containing radar datasets.
        append_dim (str): The name of the dimension to add.

    Returns:
        DataTree: A DataTree with the specified dimension added to all datasets.
    """
    start_time = pd.to_datetime(dtree.time_coverage_start.item())
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(None)

    for node in dtree.subtree:
        ds = node.to_dataset(inherit=False)
        ds[append_dim] = start_time
        ds[append_dim].attrs = {
            "description": "Volume Coverage Pattern time since start of volume scan"
        }
        ds = ds.set_coords(append_dim).expand_dims(dim=append_dim, axis=0)
        dtree[node.path].ds = ds

    return dtree


def ensure_dimension(dtree: DataTree, append_dim: str) -> DataTree:
    """
    Ensure that all datasets in a DataTree have a specified dimension.

    Parameters:
        dtree (DataTree): The DataTree to check and modify.
        append_dim (str): The name of the dimension to ensure.

    Returns:
        DataTree: The modified DataTree with the required dimension.
    """
    needs_expansion = not any(append_dim in node.dims for node in dtree.subtree)
    if needs_expansion:
        return exp_dim(dtree, append_dim)
    return dtree
