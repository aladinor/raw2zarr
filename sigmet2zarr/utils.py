#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import xarray as xr
import xradar as xd
import fsspec
import numpy as np
from datetime import datetime, timezone
from datatree import DataTree
import pandas as pd
import tomllib


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
        if not pd.isnull(time):
            return time


def fix_angle(ds: xr.Dataset, tolerance: float = None, **kwargs) -> xr.Dataset:
    """
    This function reindex the radar azimuth angle to make all sweeps starts and end at the same angle
    @param tolerance: Tolerance at which angle reindex will be performed
    @param ds: xarray dataset containing and xradar object
    @return: azimuth reindex xarray dataset
    """
    angle_dict = xd.util.extract_angle_parameters(ds)
    start_ang = angle_dict["start_angle"]
    stop_ang = angle_dict["stop_angle"]
    direction = angle_dict["direction"]
    ds = xd.util.remove_duplicate_rays(ds)
    az = len(np.arange(start_ang, stop_ang))
    ar = np.round(az / len(ds.azimuth.data), 2)
    tolerance = ar if not tolerance else tolerance
    return xd.util.reindex_angle(
        ds,
        start_ang,
        stop_ang,
        ar,
        direction,
        method="nearest",
        tolerance=tolerance,
        **kwargs,
    )


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
        groups = [i for i in list(dtree.groups) if i.startswith("/sweep_")]
        for group in groups:
            encoding.update({f"{group}": {f"{append_dim}": enc, "time": enc}})
        return encoding
    else:
        encoding.update(
            {
                f"{append_dim}": enc,
                "time": enc,
            }
        )
        return encoding
