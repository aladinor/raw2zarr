#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import xarray as xr
import xradar as xd
import fsspec
import numpy as np
from datetime import datetime, timezone
import pandas as pd
import tomllib
from time import time
from collections.abc import Iterator
from typing import Any, List
from xarray import DataTree
import gzip
import bz2


def batch(iterable: List[Any], n: int = 1) -> Iterator[List[Any]]:
    """
    Splits a list into consecutive chunks of size `n`.

    This function takes a list and yields successive batches of size `n` from it.
    If the length of the list is not evenly divisible by `n`, the last batch will
    contain the remaining elements.

    Parameters
    ----------
    iterable : list[Any]
        The list to be split into batches.
    n : int, optional
        The number of items in each batch (default is 1).

    Yields
    ------
    Iterator[list[Any]]
        An iterator that yields slices of the original list of size `n`, except
        for the last batch which may contain fewer elements if the total number
        of elements in the list is not evenly divisible by `n`.

    Examples
    --------
    >>> list(batch([1, 2, 3, 4, 5], n=2))
    [[1, 2], [3, 4], [5]]

    >>> list(batch(['a', 'b', 'c', 'd'], n=3))
    [['a', 'b', 'c'], ['d']]
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def make_dir(path) -> None:
    """
    Makes directory based on path.
    :param path: directory path that will be created
    :return:
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def load_toml(filepath: str) -> dict:
    """
    Load a TOML data from file
    @param filepath: path to TOML file
    @return: dict
    """
    with open(filepath, "rb") as f:
        toml_data: dict = tomllib.load(f)
        return toml_data


def time_3d(time_array, numbers) -> np.ndarray:
    """
    Functions that creates a 3d time array from timestamps
    :param time_array: 2d timestamp array
    :param numbers: number of times in the new axis
    :return: 3d time array
    """
    v_func = np.vectorize(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
    _time = v_func(time_array)
    times = np.repeat(_time[np.newaxis, :], numbers, axis=0)
    return times


def get_time(time_array) -> np.ndarray:
    """
    Functions that creates a 3d time array from timestamps
    :param time_array: 2d timestamp array
    :return: 3d time array
    """
    v_func = np.vectorize(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
    _time = v_func(time_array)
    return _time


def create_query(date, radar_site) -> str:
    """
    Creates a string for quering the IDEAM radar files stored in AWS bucket
    :param date: date to be queried. e.g datetime(2021, 10, 3, 12). Datetime python object
    :param radar_site: radar site e.g. Guaviare
    :return: string with a IDEAM radar bucket format
    """
    if (date.hour != 0) and (date.hour != 0):
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d%H}"
    elif (date.hour != 0) and (date.hour == 0):
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}"
    else:
        return f"l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}"


def data_accessor(file: str):
    """
    Open remotely a AWS S3 file using fsspec
    """
    with fsspec.open(file, mode="rb", anon=True) as f:
        return f.read()


def convert_time(ds) -> pd.to_datetime:
    """
    Functions that create a timestamps for appending sweep data along a given dimension
    @param ds: Xarray dataset
    @return: pandas datetime
    """
    for i in ds.time.values:
        time = pd.to_datetime(i)
        if pd.isnull(time):
            continue
        return time


def check_if_exist(file: str, path: str = "../results") -> bool:
    """
    Function that check if a sigmet file was already processed based on a txt file that written during the conversion
    @param file: file name
    @param path: path where txt file was written with the list of sigmet files processed
    @return:
    """
    file_path = f"{path}"
    file_name = f"{file_path}/{file.split('/')[-2]}_files.txt"
    try:
        with open(file_name, "r", newline="\n") as txt_file:
            lines = txt_file.readlines()
            txt_file.close()
        _file = [i for i in lines if i.replace("\n", "") == file]
        if len(_file) > 0:
            print("File already processed")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def write_file_radar(file: str, path: str = f"../results") -> None:
    """
    Write a new line with the radar filename converted. This is intended to create a checklist to avoid file
    reprocessing
    @param path: path where the txt file will be saved
    @param file: radar filename
    @return:
    """
    make_dir(path)
    file_name = f"{path}/{file.split('/')[-2]}_files.txt"
    with open(file_name, "a") as txt_file:
        txt_file.write(f"{file}\n")
        txt_file.close()


def dtree_encoding(dtree, append_dim) -> dict:
    """
    Function that creates encoding for time, append_dim, and all variables in datasets within the DataTree.

    Parameters:
        dtree (DataTree): Input xarray DataTree.
        append_dim (str): The name of the dimension to encode (e.g., "vcp_time").

    Returns:
        dict: A dictionary with encoding parameters for variables and coordinates.
    """

    _encoding = {}
    # Base encoding for time-related variables
    time_enc = dict(
        units="nanoseconds since 1950-01-01T00:00:00.00",
        dtype="int64",
        _FillValue=-9999,
    )
    # Base encoding for general variables
    var_enc = dict(
        _FillValue=-9999,
    )

    if isinstance(dtree, DataTree):
        # Append_dim encoding for each group
        _encoding = {
            f"{dtree[group].path}": {f"{append_dim}": time_enc}
            for group in dtree.groups
        }

        # Add encoding for sweeps (time and append_dim)
        for node in dtree.match("*/sweep_*").groups[2:]:
            for var_name in dtree[node].to_dataset().data_vars:
                _encoding[node].update({var_name: var_enc, "time": time_enc})

        # Remove root encoding if present
        _encoding.pop("/", None)

    else:
        _encoding = {}

    return _encoding


def prepare_for_read(filename, storage_options={"anon": True}):
    """
    Return a file-like object ready for reading.

    Open a file for reading in binary mode with transparent decompression of
    Gzip and BZip2 files. Supports local and S3 files. The resulting file-like
    object should be closed.

    Parameters
    ----------
    filename : str or file-like object
        Filename or file-like object which will be opened. File-like objects
        will not be examined for compressed data.

    Returns
    -------
    file_like : file-like object
        File-like object from which data can be read.
        @param storage_options:
    """
    # If already a file-like object, return as-is
    if hasattr(filename, "read"):
        return filename

    # Check if S3 path, and open with fsspec
    if filename.startswith("s3://"):
        # with fsspec.open(filename, mode="rb", anon=True) as f:
        #     return f.read()
        return fsspec.open(
            filename, mode="rb", compression="infer", **storage_options
        ).open()
    else:
        # Open a local file and read the first few bytes to check for compression
        file = open(filename, "rb")

    # Read first few bytes to check for compression (only for local files)
    magic = file.read(3)
    file.seek(0)  # Reset pointer to beginning after reading header

    # Detect and handle gzip compression
    if magic.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(fileobj=file)

    # Detect and handle bzip2 compression
    if magic.startswith(b"BZh"):
        return bz2.BZ2File(fileobj=file)

    # Return the file object as-is if no compression detected
    return file


def exp_dim(dt: xr.DataTree, append_dim: str) -> xr.DataTree:
    """
    Add a new dimension to all datasets in a DataTree and initialize it with a specific value.

    This function adds a new dimension to each dataset in the DataTree. The dimension is
    initialized with the `time_coverage_start` value from the root of the DataTree and
    is added as a coordinate. The new dimension is also expanded to allow additional values.

    Parameters:
        dt (xr.DataTree): The DataTree containing radar datasets.
        append_dim (str): The name of the dimension to add.

    Returns:
        xr.DataTree: A DataTree with the specified dimension added to all datasets.

    Notes:
        - The new dimension is initialized with the `time_coverage_start` attribute.
        - Attributes describing the new dimension are added for metadata.
        - The datasets are updated in place within the DataTree.

    Example:
        Add a "vcp_time" dimension to all datasets in a DataTree:

        >>> dt = exp_dim(dt, "vcp_time")
    """
    # Get the start time from the root node
    start_time = pd.to_datetime(dt.time_coverage_start.item())

    # Iterate over all nodes in the DataTree
    for node in dt.subtree:
        ds = node.to_dataset(inherit=False)  # Extract the dataset without inheritance

        # Add the new dimension with the start_time value
        ds[append_dim] = start_time

        # Define attributes for the new dimension
        attrs = {
            "description": "Volume Coverage Pattern time since start of volume scan",
        }
        ds[append_dim].attrs = attrs

        # Set the new variable as a coordinate and expand the dimension
        ds = ds.set_coords(append_dim).expand_dims(dim=append_dim, axis=0)

        # Update the dataset in the DataTree node
        dt[node.path].ds = ds

    return dt


def ensure_dimension(dt: xr.DataTree, dim: str) -> xr.DataTree:
    """
    Ensure that all datasets in a DataTree have a specified dimension.

    This function checks each dataset in the DataTree for the presence of the given dimension.
    If the dimension is missing in a dataset, it is added using expand_dims.

    Parameters:
        dt (xr.DataTree): The DataTree to check and modify.
        dim (str): The name of the dimension to ensure in each dataset.

    Returns:
        xr.DataTree: A DataTree where all datasets have the specified dimension.

    Notes:
        - If the dimension is already present in a dataset, it is left unchanged.
        - The new dimension is added without any associated coordinates.
        - This function modifies datasets in-place within the DataTree.

    Example:
        Ensure that all datasets in the DataTree have a "vcp_time" dimension:

        >>> dt = ensure_dimension(dt, "vcp_time")
    """
    dims = [node.dims for node in dt.subtree if dim in node.dims]
    if not dims:
        return exp_dim(dt, dim)
    return dt


def fix_angle(dt: xr.DataTree, tolerance: float = None, **kwargs) -> xr.DataTree:
    """
    Reindex the radar azimuth angle to ensure all sweeps start and end at the same angle.

    This function processes each sweep in a radar dataset stored in an xarray DataTree.
    It ensures that the azimuth angles are reindexed to cover a consistent range, removing
    duplicates and interpolating as needed to maintain uniform spacing.

    Parameters:
        dt (xr.DataTree): The input DataTree containing radar data, with each sweep represented as a node.
        tolerance (float, optional): Tolerance for angle reindexing. If not specified, it will be
            calculated based on the angular resolution.
        **kwargs: Additional arguments passed to `xd.util.reindex_angle`.

    Returns:
        xr.DataTree: The DataTree with azimuth angles reindexed for all sweeps.

    Notes:
        - The function assumes the nodes of interest are named using the "sweep_*" pattern.
        - It uses xradar utilities to extract angle parameters, remove duplicate rays,
          and reindex angles for uniform azimuth coverage.
        - The angular resolution (`ar`) is calculated dynamically based on the azimuth range and size.
    """
    for node in dt.match("sweep_*"):
        ds = dt[node].to_dataset()
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
        dt[node].ds = ds
    return dt


def fix_azimuth(
    ds: xr.Dataset, target_size: int = 360, fill_value: str = "extrapolate"
) -> xr.Dataset:
    """
    Ensure that the azimuth dimension in a radar dataset matches a target size.

    This function adjusts the azimuth dimension of a radar dataset to match a specified target size
    (e.g., 360 for a full sweep). It detects the starting and stopping angles of the azimuth and
    interpolates data to create a uniform azimuth grid.

    Parameters:
        ds (xr.Dataset): The dataset containing radar data with an azimuth coordinate.
        target_size (int, optional): The desired size of the azimuth dimension (default is 360).
        fill_value (str, optional): Value used to fill points outside the data range during
            interpolation. Options include:
            - "extrapolate": Fill using extrapolation (default).
            - None: Introduce `NaN` for points outside the data range.

    Returns:
        xr.Dataset: The dataset with a completed and uniformly spaced azimuth dimension.

    Notes:
        - If the current azimuth size matches the target size, no changes are made.
        - The interpolation uses `xarray.interp` with the specified `fill_value` behavior.
        - The azimuth grid is reconstructed to span from the detected start angle to the stop angle.
    """
    # Current azimuth size and coordinates
    current_size = ds["azimuth"].shape[0]
    azimuth = ds["azimuth"].values

    # Detect start and stop angles from the azimuth
    start_ang = np.min(azimuth)
    stop_ang = np.max(azimuth)

    # Check if the azimuth size needs fixing
    if current_size % target_size != 0:
        print(f"Fixing azimuth dimension from {current_size} to {target_size}")
        time_numeric = ds["time"].astype("float64")
        # Create a new uniform azimuth array within the detected range
        new_azimuth = np.linspace(start_ang, stop_ang, target_size, endpoint=False)

        # Interpolate data to the new azimuth array
        ds = ds.interp(azimuth=new_azimuth, kwargs={"fill_value": fill_value})
        # Interpolate the `time` coordinate explicitly if it exists
        if "time" not in ds.coords:
            # Convert datetime64 to numeric
            time_interpolated = xr.DataArray(
                pd.to_datetime(np.interp(new_azimuth, azimuth, time_numeric)),
                dims="azimuth",
                coords={"azimuth": new_azimuth},
            )
            ds["time"] = time_interpolated
            ds = ds.set_coords("time")

    return ds
