from xarray import DataTree

from ..templates.template_manager import VcpTemplateManager


def get_vcp_values(vcp_name: str = "VCP-212") -> list[float]:
    """
    Load elevation angles for a given Volume Coverage Pattern (VCP).

    Parameters:
        vcp_name (str): Name of the VCP (e.g., "VCP-212").

    Returns:
        list[float]: Elevation angles in degrees.

    Raises:
        KeyError: If the VCP name is not found.
        ValueError: If the structure is invalid.
    """
    template_mgr = VcpTemplateManager()

    try:
        vcp_info = template_mgr.get_vcp_info(vcp_name)
        elevations = vcp_info.elevations
    except ValueError as e:
        raise KeyError(f"VCP '{vcp_name}' not found in unified config.") from e

    if not isinstance(elevations, list) or not all(
        isinstance(e, (int, float)) for e in elevations
    ):
        raise ValueError(f"Invalid 'elevations' list for {vcp_name}: {elevations}")

    return elevations


def _get_missing_elevations(dtree: DataTree, tolerance: float = 0.05) -> list[float]:
    vcp = dtree.attrs["scan_name"]
    default_sweeps = get_vcp_values(vcp_name=vcp)
    actual_sweeps = [
        dtree[path].ds["sweep_fixed_angle"].values.item()
        for path in dtree.match("sweep_*").children
    ]
    i = 0
    j = 0
    while i < len(default_sweeps) and j < len(actual_sweeps):
        if abs(default_sweeps[i] - actual_sweeps[j]) <= tolerance:
            i += 1
            j += 1
        else:
            j += 1
    return [idx for idx in range(i, len(default_sweeps))]


def create_empty_vcp_datatree(vcp_id: str, radar_info: dict) -> DataTree:
    """
    Create a DataTree with empty datasets for all scans in a VCP

    Parameters:
        vcp_id: Volume Coverage Pattern ID (e.g., "VCP-21")
        radar_info: Dictionary with radar metadata:
            - lon: Radar longitude
            - lat: Radar latitude
            - alt: Radar altitude
            - reference_time: Volume start time
            - vcp_time: VCP timestamp

    Returns:
        DataTree: Hierarchical structure with empty scans for all expected elevations
    """
    template_mgr = VcpTemplateManager()

    # Update radar_info to include VCP
    vcp_radar_info = {**radar_info, "vcp": vcp_id}

    # Use unified config system to create template
    vcp_info = template_mgr.get_vcp_info(vcp_id)
    empty_datasets = {}

    for idx in range(len(vcp_info.elevations)):
        empty_ds = template_mgr.create_scan_dataset(
            scan_type=f"unified_sweep_{idx}",
            sweep_idx=idx,
            radar_info=vcp_radar_info,
        )
        node_name = f"sweep_{idx}"
        empty_datasets[node_name] = empty_ds

    return DataTree.from_dict(empty_datasets)


def get_root_datasets(dtree: DataTree) -> dict:
    rad_params = "radar_parameters"
    geo_corr = "georeferencing_correction"
    rad_cal = "radar_calibration"
    root_dict = {
        "/": dtree.root.to_dataset(),
        rad_params: dtree[rad_params].to_dataset(),
        geo_corr: dtree[geo_corr].to_dataset(),
        rad_cal: dtree[rad_cal].to_dataset(),
    }
    return root_dict
