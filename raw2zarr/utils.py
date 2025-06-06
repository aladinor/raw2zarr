#!/usr/bin/env python
import bz2
import gzip
import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import tomllib
import xarray as xr
import xradar as xd
from xarray import DataTree


def batch(iterable: list[Any], n: int = 1) -> Iterator[list[Any]]:
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
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def timer_func(func):
    """Decorator that prints the execution time in h:m:s."""

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        elapsed = t2 - t1

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        print(
            f"Function {func.__name__!r} executed in {hours}h {minutes}m {seconds:.2f}s"
        )
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
        with open(file_name, newline="\n") as txt_file:
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


def write_file_radar(file: str, path: str = "../results") -> None:
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
    from collections import defaultdict

    time_enc = dict(
        units="nanoseconds since 1950-01-01T00:00:00.00",
        dtype="int64",
        _FillValue=-9999,
    )
    var_enc = dict(
        _FillValue=-9999,
    )
    encoding = defaultdict(dict)
    if not isinstance(dtree, DataTree):
        return {}

    for node in dtree.subtree:
        path = node.path
        ds = node.ds
        if node.is_empty:
            continue
        if "time" in ds:
            encoding[path]["time"] = time_enc
        if append_dim in ds:
            encoding[path][append_dim] = time_enc
        # will trigger warning. Still waiting for https://github.com/pydata/xarray/issues/10077
        for var_name, var in ds.data_vars.items():
            if var.dtype.kind in {"O", "U"}:
                encoding[path][var_name] = {
                    "dtype": np.dtypes.StringDType,
                }
            else:
                encoding[path][var_name] = var_enc

    return dict(encoding)


def prepare2read(filename: str, storage_options: dict = None):
    """
    Return a file-like object ready for reading.

    Opens a file for reading in binary mode with automatic detection and handling of
    Gzip and BZip2 compressed files. Supports local and S3 files through `fsspec`.
    If a file is hosted on S3, it uses the given storage options for authentication.
    The caller is responsible for closing the returned file-like object.

    Parameters
    ----------
    filename : str or file-like object
        The path to the file to be opened, either as a string (for local or S3 files)
        or an already open file-like object. If the input is a file-like object, it will be returned as-is.
    storage_options : dict, optional
        A dictionary containing authentication options for S3. By default,
        it uses anonymous authentication (`{"anon": True}`). This parameter is ignored for local files.

    Returns
    -------
    file-like object
        A file-like object that can be used to read the contents of the specified file.
        For compressed files, the appropriate decompression object (gzip or bzip2) is returned.
        For S3 files, the object returned by `fsspec.open` is used. Otherwise, the standard
        file object from `open()` is returned.

    Raises
    ------
    FileNotFoundError
        If the specified local file does not exist.
    ValueError
        If the filename string is not a valid S3 path when using S3 storage options.
    fsspec.exceptions.FSSpecError
        For errors related to S3 file handling.

    Examples
    --------
    Open a local file for reading:
    >>> with prepare2read("example.txt") as f:
    ...     data = f.read()

    Open a gzip-compressed file:
    >>> with prepare2read("example.txt.gz") as f:
    ...     data = f.read()

    Open an S3 file with anonymous access:
    >>> with prepare2read("s3://mybucket/example.txt") as f:
    ...     data = f.read()

    Open an S3 file with specific storage options:
    >>> storage_options = {"anon": False, "key": "my-access-key", "secret": "my-secret-key"}
    >>> with prepare2read("s3://mybucket/example.txt", storage_options=storage_options) as f:
    ...     data = f.read()
    """
    if not storage_options:
        storage_options = {"anon": True}
    # If already a file-like object, return as-is
    if hasattr(filename, "read"):
        return filename

    # Check if S3 path, and open with fsspec
    if filename.startswith("s3://"):
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


def exp_dim(dtree: xr.DataTree, append_dim: str) -> xr.DataTree:
    """
    Add a new dimension to all datasets in a DataTree and initialize it with a specific value.

    This function adds a new dimension to each dataset in the DataTree. The dimension is
    initialized with the `time_coverage_start` value from the root of the DataTree and
    is added as a coordinate. The new dimension is also expanded to allow additional values.

    Parameters:
        dtree (xr.DataTree): The DataTree containing radar datasets.
        append_dim (str): The name of the dimension to add.

    Returns:
        xr.DataTree: A DataTree with the specified dimension added to all datasets.

    Notes:
        - The new dimension is initialized with the `time_coverage_start` attribute.
        - Attributes describing the new dimension are added for metadata.
        - The datasets are updated in place within the DataTree.

    Example:
        Add a "vcp_time" dimension to all datasets in a DataTree:

        >>> dt = exp_dim(dtree, "vcp_time")
    """
    start_time = pd.to_datetime(dtree.time_coverage_start.item())
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(None)

    for node in dtree.subtree:
        ds = node.to_dataset(inherit=False)  # Extract the dataset without inheritance
        ds[append_dim] = start_time
        attrs = {
            "description": "Volume Coverage Pattern time since start of volume scan",
        }
        ds[append_dim].attrs = attrs
        ds = ds.set_coords(append_dim).expand_dims(dim=append_dim, axis=0)
        dtree[node.path].ds = ds

    return dtree


def ensure_dimension(dt: xr.DataTree, append_dim: str) -> xr.DataTree:
    """
    Ensure that all datasets in a DataTree have a specified dimension.

    This function checks each dataset in the DataTree for the presence of the given dimension.
    If the dimension is missing in a dataset, it is added using expand_dims.

    Parameters:
        dt (xr.DataTree): The DataTree to check and modify.
        append_dim (str): The name of the dimension to ensure in each dataset.

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
    dims = [node.dims for node in dt.subtree if append_dim in node.dims]
    if not dims:
        return exp_dim(dt, append_dim)
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


def fix_azimuth(ds: xr.Dataset, fill_value: str = "extrapolate") -> xr.Dataset:
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

    target_size = round(current_size / 360) * 360
    if target_size < 360:
        target_size = 360  # Ensure minimum target size is 360

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


def load_json_config(filename: str = "vcp.json") -> dict:
    """
    Load and parse a JSON configuration file from the correct package path.

    Parameters:
        filename (str): Name of the JSON file (e.g., 'vcp.json', 'scan_config.json').

    Returns:
        dict: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file is not found at the expected location.
        ValueError: If the JSON file contains invalid syntax.
    """
    try:
        # Adjust base directory to point to 'raw2zarr' package root
        package_root = Path(__file__).resolve().parent  # Get the raw2zarr directory

        # Define the correct path inside raw2zarr/config/
        config_path = package_root / "config" / filename

        if not config_path.exists():
            raise FileNotFoundError(f"{filename} not found at {config_path}")

        # Read and parse the JSON file
        with config_path.open("r", encoding="utf-8") as file:
            config_data = json.load(file)

        return config_data
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{filename} not found at expected location: {config_path}"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {filename}: {e}")


def get_vcp_values(vcp_name: str = "VCP-212") -> list[float]:
    """
    Retrieve VCP values from the loaded TOML config.

    Args:
        vcp_name (str): The specific VCP name (e.g., 'VCP-212').

    Returns:
        list[float]: The list of elevation angles for the given VCP.

    Raises:
        KeyError: If the engine or VCP name is not found in the config.
    """
    config = load_json_config("vcp.json")  # Load the configuration
    try:
        values = config[vcp_name]["elevations"]
        if not isinstance(values, list):
            raise ValueError(
                f"Expected a list of floats for {vcp_name}, got {type(values).__name__}"
            )
        return values
    except KeyError as e:
        raise KeyError(f"VCP '{vcp_name}' not found.") from e


def _get_missing_elevations(
    default_list: list, second_list: list, tolerance: float = 0.05
) -> list[float]:
    i = 0  # Index for default_list
    j = 0  # Index for second_list
    while i < len(default_list) and j < len(second_list):
        if abs(default_list[i] - second_list[j]) <= tolerance:
            # Match: move both indices
            i += 1
            j += 1
        else:
            # Mismatch: skip the current element in the second list
            j += 1

    # Return indexes of remaining elements in the default list
    return [idx for idx in range(i, len(default_list))]


def check_dynamic_scan(dtree: xr.DataTree) -> bool:
    """
    Checks for the presence of adaptive scanning in radar data represented by a xarray DataTree.

    Adaptive scanning in radar operations can involve repeated sweeps at the same elevation
    angle, often due to techniques like SAILS or MRLE. This function determines whether
    adaptive scanning is present by analyzing the frequency of repeated elevation angles
    in the DataTree.

    Parameters:
        dtree (xr.DataTree):
            An xarray DataTree containing radar data, where each sweep is represented
            as a node with a `sweep_fixed_angle` attribute.

    Returns:
        bool:
            `True` if adaptive scanning is detected (elevation angle is repeated more
            than twice), otherwise `False`.

    Algorithm:
        - Extracts the `sweep_fixed_angle` for all sweeps in the DataTree.
        - Counts the occurrences of each unique elevation angle.
        - If any elevation angle occurs more than twice, adaptive scanning is inferred.

    Example:
        >>> from xarray import DataTree
        >>> dtree = DataTree(...)  # Load your radar data into a DataTree
        >>> is_adaptive = check_dynamic_scan(dtree)
        >>> print(is_adaptive)  # Outputs True or False based on the scan pattern.
    """
    current_vcp = dtree.attrs["scan_name"]
    elevations: list[float] = [
        dtree[sweep]["sweep_fixed_angle"].item() for sweep in dtree.match("*sweep_*")
    ]
    VCP_REFERENCE = get_vcp_values(vcp_name=current_vcp)

    if not VCP_REFERENCE:
        raise ValueError(
            f"VCP type '{current_vcp}' not found in the reference dictionary."
        )

    # Check if the lengths of the lists match
    if len(elevations) != len(VCP_REFERENCE):
        return True

    # Compare each angle with the reference, allowing for a small tolerance
    for extracted, reference in zip(elevations, VCP_REFERENCE):
        if abs(extracted - reference) > 0.05:
            return True

    return False


def dtree_full_like(dtree: DataTree, fill_value="NaN") -> DataTree:
    if not isinstance(dtree, DataTree):
        raise TypeError(
            f"Please pass a DataTree. dtree_full_like does not support {type(dtree)}"
        )

    # Use subtree to iterate over all nodes
    new_tree = {
        node.path: (
            xr.full_like(node.ds, fill_value=fill_value)
            if node.ds is not None
            else None
        )
        for node in dtree.subtree
    }
    return DataTree.from_dict(new_tree)


def create_empty_dtree(vcp: str = "VCP-212") -> DataTree:
    from raw2zarr.dtree_builder import align_dynamic_scan, load_radar_data

    if vcp == "VCP-212":
        file = "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_161526_V06"
    engine = "nexradlevel2"
    append_dim = "vcp_time"
    dtree = load_radar_data(file, engine=engine)
    dtree = (dtree.pipe(fix_angle)).xradar.georeference()
    if (engine == "nexradlevel2") & check_dynamic_scan(dtree):
        dtree = align_dynamic_scan(dtree, append_dim=append_dim, engine=engine)
    else:
        dtree = dtree.pipe(ensure_dimension, append_dim)
    return dtree_full_like(dtree.isel(vcp_time=0), np.nan)
