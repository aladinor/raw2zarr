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
from xarray import DataTree, Dataset
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


def time_encoding(dtree, append_dim) -> dict:
    """
    Function that creates encoding for time and append_dim variables
    @param dtree: Input xarray Datatree
    @param append_dim: dimension name. e.g. "vcp_time"
    @return: dict with encoding parameters
    """
    encoding = {}
    enc = dict(
        units="nanoseconds since 2000-01-01T00:00:00.00",
        dtype="float64",
        _FillValue=np.datetime64("NaT"),
    )
    if type(dtree) is DataTree:
        #  [dtree[node].data_vars for node in dtree.match("sweep_*")]
        encoding = (
            {
                f"{group.parent.path}": {f"{append_dim}": enc}  # , "time": enc}
                for group in dtree.match("sweep_*").leaves
                if append_dim in list(group.variables)
            }
            if isinstance(dtree, DataTree)
            else {}
        )
        encoding.update(
            {f"{group.path}": {"time": enc} for group in dtree.match("sweep_*").leaves}
        )
        return encoding
    else:
        encoding.update(
            {
                f"{append_dim}": enc,
                "time": enc,
            }
        )
        return encoding


def prepare_for_read(filename):
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
    """
    # If already a file-like object, return as-is
    if hasattr(filename, "read"):
        return filename

    # Check if S3 path, and open with fsspec
    if filename.startswith("s3://"):
        with fsspec.open(filename, mode="rb", anon=True) as f:
            return f.read()
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


def exp_dim(dt: DataTree, append_dim: str):
    try:
        start_time = dt.time_coverage_start.item()
    except ValueError as e:
        print(e)
    for node in dt.subtree:
        ds = node.to_dataset()
        ds[append_dim] = pd.to_datetime(start_time)
        ds = ds.set_coords(append_dim).expand_dims(dim=append_dim, axis=0)
        dt[node.path].ds = ds
    return dt


def ensure_dimension(dt: xr.DataTree, dim: str) -> xr.Dataset:
    """
    Ensure that a dataset has a given dimension. If the dimension is missing,
    add it using expand_dims with the specified coordinate value.

    Parameters:
        ds (xr.Dataset): The dataset to check.
        dim (str): The name of the dimension to ensure.
        coord_value: The coordinate value to use if adding the dimension.

    Returns:
        xr.Dataset: Dataset with the specified dimension.
    """
    dims = [node.dims for node in dt.subtree if dim in node.dims]
    if not dims:
        return exp_dim(dt, dim)
    return dt


def fix_angle(dt: xr.DataTree, tolerance: float = None, **kwargs) -> xr.DataTree:
    """
    This function reindex the radar azimuth angle to make all sweeps starts and end at the same angle
    @param tolerance: Torelance for angle reindex.
    @param ds: xarray dataset containing and xradar object
    @return: azimuth reindex xarray dataset
    """
    for node in dt.match("sweep_*"):
        ds = dt[node].to_dataset()
        ds["time"] = ds.time.load()
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
