from itertools import zip_longest
from math import isclose

import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from xarray import Dataset, DataTree

from .utils import get_vcp_values


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
            continue  # Skip malformed or missing sweep_fixed_angle

    if len(extracted_elevs) != len(reference_elevs):
        return True

    for a, b in zip_longest(extracted_elevs, reference_elevs, fillvalue=None):
        if a is None or b is None or not isclose(a, b, abs_tol=tolerance):
            return True

    return False
