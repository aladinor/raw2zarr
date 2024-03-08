#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from configparser import ConfigParser
from datetime import datetime


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


if __name__ == '__main__':
    pass
