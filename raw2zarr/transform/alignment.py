from itertools import zip_longest
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from xarray import Dataset, DataTree

from ..templates.template_manager import ScanTemplateManager
from ..transform.dimension import ensure_dimension
from ..transform.utils import load_json_config
from .utils import (
    _get_missing_elevations,
    create_empty_vcp_datatree,
    get_root_datasets,
    get_vcp_values,
)


def reindex_angle(ds: Dataset, tolerance: float = None) -> Dataset:
    ds = ds.copy(deep=True)
    ds["time"] = ds.time.load()
    ds = fix_azimuth(ds)
    angle_dict = xd.util.extract_angle_parameters(ds)
    start_ang = angle_dict["start_angle"]
    stop_ang = angle_dict["stop_angle"]
    direction = angle_dict["direction"]
    ds = xd.util.remove_duplicate_rays(ds)
    az = len(np.arange(start_ang, stop_ang))
    ar = np.round(az / len(ds.azimuth.data), 2)
    tolerance = ar if not tolerance else tolerance
    ds = xd.util.reindex_angle(
        ds,
        start_ang,
        stop_ang,
        ar,
        direction,
        method="nearest",
        tolerance=tolerance,
    )
    return ds


def fix_angle(dtree: DataTree, tolerance: float = None, **kwargs) -> DataTree:
    """
    Reindexes all sweeps in a radar DataTree to consistent azimuth angles.

    Parameters
    ----------
    dtree : xarray.DataTree
        DataTree with radar sweep nodes (e.g., "sweep_0", "sweep_1", ...)
    tolerance : float, optional
        Tolerance for reindexing angles. If not provided, it is estimated from sweep resolution.
    **kwargs : dict
        Extra arguments passed to `xradar.util.reindex_angle`.

    Returns
    -------
    xarray.DataTree
        Updated DataTree with azimuth angles realigned and interpolated as needed.

    Notes
    -----
    - Uses `xradar.util.extract_angle_parameters` to determine sweep geometry.
    - Handles duplicate rays and non-uniform angle spacing.
    - Modifies `dt` in-place.
    """
    # TODO: this should work for single sweeps as well
    return dtree.xradar.map_over_sweeps(
        reindex_angle,
        tolerance=tolerance,
        **kwargs,
    )


def fix_azimuth(ds: Dataset, fill_value: str = "extrapolate") -> Dataset:
    """
    Interpolates a radar sweep to a uniform azimuth grid.

    Parameters
    ----------
    ds : xarray.Dataset
        A radar sweep dataset containing an "azimuth" coordinate.
    fill_value : str, optional
        How to fill values outside the interpolation range:
        - "extrapolate" (default) to fill using edge values
        - None to leave gaps as NaN

    Returns
    -------
    xr.Dataset
        Dataset with interpolated azimuth grid and (optionally) interpolated time coordinate.

    Notes
    -----
    - Ensures azimuth resolution is uniform (e.g., 360 bins).
    - If azimuth size is already standard, the dataset is returned unchanged.
    - Time coordinate is interpolated separately if missing.
    """
    current_size = ds["azimuth"].shape[0]
    azimuth = ds["azimuth"].values

    start_ang = np.min(azimuth)
    stop_ang = np.max(azimuth)

    target_size = round(current_size / 360) * 360
    if target_size < 360:
        target_size = 360

    if current_size % target_size != 0:
        time_numeric = ds["time"].astype("float64")
        new_azimuth = np.linspace(start_ang, stop_ang, target_size, endpoint=False)

        ds = ds.interp(azimuth=new_azimuth, kwargs={"fill_value": fill_value})
        if "time" not in ds.coords:
            time_interpolated = xr.DataArray(
                pd.to_datetime(np.interp(new_azimuth, azimuth, time_numeric)),
                dims="azimuth",
                coords={"azimuth": new_azimuth},
            )
            ds["time"] = time_interpolated
            ds = ds.set_coords("time")
    return ds


def check_dynamic_scan(dtree: xr.DataTree, tolerance: float = 0.05) -> bool:
    """
    Determine if a radar scan uses adaptive scanning (e.g., SAILS/MRLE) by comparing
    its sweep elevations with expected VCP configuration.

    Parameters:
        dtree (xr.DataTree): Radar DataTree with sweep_* nodes.
        tolerance (float): Allowed deviation in degrees when comparing angles.

    Returns:
        bool: True if adaptive scanning is detected (missing or repeated sweeps).
    """
    scan_name = dtree.attrs.get("scan_name")
    if not scan_name:
        raise ValueError("Missing 'scan_name' attribute in DataTree root.")

    try:
        reference_elevs = get_vcp_values(scan_name)
    except KeyError:
        raise ValueError(f"VCP reference not found for '{scan_name}'.")

    extracted_elevs = []
    for node in dtree.match("sweep_*").values():
        try:
            elev = node["sweep_fixed_angle"].values.item()
            extracted_elevs.append(float(elev))
        except Exception:
            continue

    if len(extracted_elevs) != len(reference_elevs):
        return True

    for a, b in zip_longest(extracted_elevs, reference_elevs, fillvalue=None):
        if a is None or b is None or not isclose(a, b, abs_tol=tolerance):
            return True

    return False


def extract_and_group_sweeps(dtree: DataTree) -> dict:
    """
    Groups sweep datasets by (fixed_angle, dims) key.

    Returns:
        dict: {(fixed_angle, dims): [(name, dataset), ...]}
    """
    sweep_groups = {}

    for node_name, node in dtree.match("sweep_*").items():
        ds = node.ds
        if "sweep_fixed_angle" not in ds:
            raise ValueError(f"'sweep_fixed_angle' missing in {node_name}")

        angle = ds["sweep_fixed_angle"].values.item()
        dims = tuple(ds.sizes.items())

        key = (angle, dims)
        sweep_groups.setdefault(key, []).append((node_name, ds))

    return sweep_groups


def build_new_dtree(
    sweep_groups: dict, append_dim: str, start_time: pd.Timestamp
) -> dict:
    """
    Construct a dictionary of aligned sweep datasets with vcp_time added.

    Parameters:
        sweep_groups (dict): A dict of {(fixed_angle, dims): [(name, xr.Dataset), ...]}
        append_dim (str): The dimension to expand along (e.g., 'vcp_time')
        start_time (pd.Timestamp): The starting timestamp for the first sweep

    Returns:
        dict: A dictionary mapping sweep_n to xarray.Dataset
    """
    new_dtree = {}
    i = 0

    for key, node_ds_list in sweep_groups.items():
        group_path, datasets = zip(*node_ds_list)
        group_path = group_path[0]
        if group_path.startswith("sweep"):
            group_path = f"sweep_{i}"
            i += 1

        if len(datasets) > 1:
            # Multiple sweeps to concatenate
            time_coords = [pd.to_datetime(ds.time.mean().values) for ds in datasets]
            time_coords[0] = start_time
            time_coords = [
                ts.tz_convert(None) if ts.tzinfo else ts for ts in time_coords
            ]

            time_coords_da = xr.DataArray(
                data=time_coords,
                dims=(append_dim,),
                name=append_dim,
                attrs={"description": "Volume Coverage Pattern time"},
            )
            concat_ds = xr.concat(datasets, dim=append_dim)
            concat_ds[append_dim] = time_coords_da
            new_dtree[group_path] = concat_ds.set_coords(append_dim)

        else:
            # Single sweep: expand along append_dim
            ds = datasets[0].copy()
            ds[append_dim] = start_time
            ds[append_dim].attrs = {"description": "Volume Coverage Pattern time"}
            ds = ds.set_coords(append_dim).expand_dims(dim=append_dim, axis=0)

            # Optionally expand 2D coords (e.g., time, elevation) to 3D
            for coord in ["time", "elevation", "x", "y", "z"]:
                if coord in ds.coords:
                    ds[coord] = ds[coord].expand_dims(dim=append_dim, axis=0)

            new_dtree[group_path] = ds

    return new_dtree


def align_dynamic_scan(dtree: DataTree, append_dim: str = "vcp_time") -> DataTree:
    vcp = dtree.attrs["scan_name"]
    missing_idx = _get_missing_elevations(dtree)
    dict_root = get_root_datasets(dtree)
    sweep_groups_dt = extract_and_group_sweeps(dtree)

    start_time = pd.to_datetime(dtree.time_coverage_start.item())
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(None)
    new_sweep_groups = build_new_dtree(
        sweep_groups=sweep_groups_dt, append_dim=append_dim, start_time=start_time
    )

    radar_info = {
        "lon": dtree.root.longitude.item(),  # Changed from radar_lon
        "lat": dtree.root.latitude.item(),  # Changed from radar_lat
        "alt": dtree.root.altitude.item(),  # Changed from radar_alt
        "reference_time": pd.to_datetime(dtree.time_coverage_start.item()).to_numpy(),
        "vcp": vcp,
    }

    new_dtree = dict_root | new_sweep_groups

    if missing_idx:
        missing_sweeps = fill_missing_sweeps(
            missing_idx=missing_idx,
            start_time=start_time,
            append_dim=append_dim,
            radar_info=radar_info,
            vcp=vcp,
        )
        new_dtree = new_dtree | missing_sweeps
    # TODO: fix the alignment of dtrees
    # new_dtree = _dtree_aligment(new_sweep_groups, append_dim=append_dim, radar_info=radar_info)
    return DataTree.from_dict(new_dtree).pipe(ensure_dimension, append_dim)


def _dtree_aligment(tree_dict: dict, append_dim: str, radar_info: dict) -> dict:
    vcp_id = radar_info.pop("vcp")
    max_vcp_time = 0
    for path, ds in tree_dict.items():
        if append_dim in ds.coords:
            len_vcp_time = len(ds.coords[append_dim].values)
            if len_vcp_time > max_vcp_time:
                max_vcp_time = len_vcp_time

    all_vcp_time_coords = set()
    for ds in tree_dict.values():
        if append_dim in ds.coords:
            all_vcp_time_coords.update(ds.coords[append_dim].values)

    # Convert the superset to a sorted array (ensure it's datetime64[ns])
    unified_time = pd.to_datetime(list(all_vcp_time_coords)[:max_vcp_time])
    for time in unified_time:
        radar_info["vcp_time"] = time
        create_empty_vcp_datatree(vcp_id=vcp_id, radar_info=radar_info)
    # Align all datasets to the unified vcp_time coordinates
    aligned_tree_dict = {}
    for path, ds in tree_dict.items():
        if append_dim in ds.coords:
            if path == "/":
                method = "pad"
            else:
                method = None
            # Reindex the dataset to include the unified coordinates, using NaT for missing values
            aligned_ds = ds.reindex({append_dim: unified_time}, method=method)
            aligned_tree_dict[path] = aligned_ds
        else:
            # If vcp_time_dim is missing, create a dataset with only the unified coordinates
            aligned_ds = xr.Dataset(coords={append_dim: unified_time})
            aligned_tree_dict[path] = aligned_ds

    return aligned_tree_dict


def fill_missing_sweeps(
    missing_idx: list[int],
    start_time: pd.Timestamp,
    append_dim: str,
    radar_info: dict,
    vcp: str,
    config_file: str = "vcp.json",
) -> dict[str, xr.Dataset]:
    """
    Generate placeholder sweep datasets for missing VCP indices using scan templates.

    Parameters:
        missing_idx (list[int]): List of missing sweep indices.
        start_time (pd.Timestamp): Time to assign to new sweeps.
        append_dim (str): Name of the dimension to use (e.g., "vcp_time").
        radar_info (dict): Metadata including lon, lat, alt, and reference_time.
        vcp (str): Volume Coverage Pattern name (e.g., "VCP-212").
        config_file (str): JSON config file containing VCP templates.

    Returns:
        dict[str, xr.Dataset]: Mapping of group paths (e.g., "sweep_3") to xarray datasets.
    """

    config_dir = Path(__file__).resolve().parent.parent / "config"
    vcp_config_path = config_dir / config_file
    scan_config_path = config_dir / "scan_config.json"

    template_mgr = ScanTemplateManager(
        scan_config_path=scan_config_path,
        vcp_config_path=vcp_config_path,
    )
    vcp_config = load_json_config(vcp_config_path)[vcp]

    filled = {}
    for idx in missing_idx:
        scan_type = vcp_config["scan_types"][idx]

        empty_ds = template_mgr.create_scan_dataset(
            scan_type=scan_type, sweep_idx=idx, radar_info=radar_info
        )

        empty_ds[append_dim] = start_time
        empty_ds[append_dim].attrs = {
            "description": "Volume Coverage Pattern time since start of volume scan"
        }

        group_path = f"sweep_{idx}"
        filled[group_path] = empty_ds.set_coords(append_dim).expand_dims(
            dim=append_dim, axis=0
        )

    return filled
