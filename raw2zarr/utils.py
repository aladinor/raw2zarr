#!/usr/bin/env python
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from time import time
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import tomllib
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
    return None


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


def create_empty_dtree(vcp: str = "VCP-212") -> DataTree:
    from .transform.alignment import align_dynamic_scan, fix_angle, check_dynamic_scan
    from .io.load import load_radar_data
    from .transform.dimension import ensure_dimension

    if vcp == "VCP-212":
        file = "s3://noaa-nexrad-level2/2023/06/29/KILX/KILX20230629_161526_V06"
    engine = "nexradlevel2"
    append_dim = "vcp_time"
    dtree = load_radar_data(file, engine=engine)
    dtree = (dtree.pipe(fix_angle)).xradar.georeference()
    if (engine == "nexradlevel2") & check_dynamic_scan(dtree):
        dtree = align_dynamic_scan(dtree, append_dim=append_dim)
    else:
        dtree = dtree.pipe(ensure_dimension, append_dim)
    return dtree_full_like(dtree.isel(vcp_time=0), np.nan)
