#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import xarray as xr
import xradar as xd
import fsspec
import numpy as np
from configparser import ConfigParser
from datetime import datetime
import pandas as pd


def make_dir(path):
    """
    Makes directory based on path.
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def get_pars_from_ini(file_name='radar'):
    """
    Returns dictionary with data for creating a xarray dataset from hdf5 file
    :param file_name: campaign from data comes from
    :type file_name: str
    :return: data from config files
    """
    file_name = f'../config/{file_name}.ini'
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(file_name)

    dt_pars = {}

    groups = parser.sections()
    for group in groups:
        db = {}
        params = parser.items(group)

        for param in params:
            try:
                db[param[0]] = eval(param[1])

            except ValueError:
                db[param[0]] = param[1].strip()

            except NameError:
                db[param[0]] = param[1].strip()

            except SyntaxError:
                db[param[0]] = param[1].strip()

        dt_pars[group] = db

    return dt_pars


def time_3d(time_array, numbers):
    """
    Functions that creates a 3d time array from timestamps
    :param time_array: 2d timestamp array
    :param numbers: number of times in the new axis
    :return: 3d time array
    """
    v_func = np.vectorize(lambda x: datetime.utcfromtimestamp(x))
    _time = v_func(time_array)
    times = np.repeat(_time[np.newaxis, :], numbers, axis=0)
    return times


def get_time(time_array):
    """
    Functions that creates a 3d time array from timestamps
    :param time_array: 2d timestamp array
    :return: 3d time array
    """
    v_func = np.vectorize(lambda x: datetime.utcfromtimestamp(x))
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
        return f'l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d%H}'
    elif (date.hour != 0) and (date.hour == 0):
        return f'l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}'
    else:
        return f'l2_data/{date:%Y}/{date:%m}/{date:%d}/{radar_site}/{radar_site[:3].upper()}{date:%y%m%d}'


def data_accessor(file) -> str:
    """
    Open AWS S3 file(s), which can be resolved locally by file caching
    """
    return fsspec.open_local(f'simplecache::s3://{file}', s3={'anon': True},
                             filecache={'cache_storage': '/tmp/radar/'})


def convert_time(dt) -> pd.to_datetime:
    return pd.to_datetime(dt.time.values[0])


def fix_angle(ds) -> xr.Dataset:
    angle_dict = xd.util.extract_angle_parameters(ds)
    start_ang = angle_dict["start_angle"]
    stop_ang = angle_dict["stop_angle"]
    direction = angle_dict["direction"]
    ds = xd.util.remove_duplicate_rays(ds)
    az = len(np.arange(start_ang, stop_ang))
    ar = az / len(ds.azimuth.data)
    tol = ar / 2
    return xd.util.reindex_angle(ds, start_ang, stop_ang, ar, direction, method="nearest", tolerance=tol)


def check_if_exist(file) -> bool:
    path = f'../results'
    file_path = f"{path}"
    file_name = f"{file_path}/{file.split('/')[-2]}_files2.txt"
    try:
        with open(file_name, 'r', newline='\n') as txt_file:
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


def write_file_radar(file) -> None:
    path = f'../results'
    file_path = f"{path}"
    make_dir(file_path)
    file_name = f"{file_path}/{file.split('/')[-2]}_files.txt"
    with open(file_name, 'a') as txt_file:
        txt_file.write(f"{file}\n")
        txt_file.close()