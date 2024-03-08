#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import fsspec
import time
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
import zarr
from datatree import DataTree
from utils import get_pars_from_ini, make_dir


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


def get_time(dt) -> pd.to_datetime:
    return pd.to_datetime(dt.time.values[0])


def raw_to_dt(file) -> DataTree:
    radar_name = file.split('/')[-1].split('.')[0][:3]
    elev = np.array(get_pars_from_ini('radar')[radar_name]['elevations'])
    swps = {j: f"sweep_{idx}" for idx, j in enumerate(elev)}
    data = {}
    dt = xd.io.open_iris_datatree(data_accessor(file))
    data.update({float(dt[j].sweep_fixed_angle.values): fix_angle(dt[j]).ds.xradar.georeference()
                 for j in list(dt.children)})
    return DataTree.from_dict({swps[k]: data[k] for k in list(data.keys())})


def dt2zarr2(dt, **kwargs) -> None:
    if kwargs['zarr_version'] == 3:
        st = zarr.DirectoryStoreV3(kwargs['store'])
    else:
        st = zarr.DirectoryStore(kwargs['store'])
    nodes = st.listdir()
    args = kwargs.copy()
    for child in list(dt.children):
        ds = dt[child].to_dataset()
        _time = get_time(ds)
        ds[kwargs['append_dim']] = _time
        ds = ds.expand_dims(dim=kwargs['append_dim'], axis=0).set_coords(kwargs['append_dim'])
        if child in nodes:
            ds.to_zarr(group=child, **args)
        else:
            try:
                args = kwargs.copy()
                del args['append_dim']
                args['mode'] = 'w-'
                encoding = {kwargs['append_dim']: {'units': 'milliseconds since 2000-01-01T00:00:04.010000',
                                                   'dtype': 'float'},
                            'time': {'units': 'milliseconds since 2000-01-01T00:00:04.010000',
                                     'dtype': 'float'}
                            }
                ds.to_zarr(group=child, encoding=encoding, **args)
            except zarr.errors.ContainsGroupError:
                args = kwargs.copy()
                ds.to_zarr(group=child, **args)


def raw2dt(file, **kwargs) -> None:
    dt = raw_to_dt(file)
    elevations = [np.round(np.median(dt.children[i].elevation.data), 1) for i in list(dt.children)]
    try:
        if kwargs['elevation'] in elevations:
            del kwargs['elevation']
            dt2zarr2(dt=dt, **kwargs)
            del dt
            write_file_radar(file)
    except KeyError:
        dt2zarr2(dt=dt, **kwargs)


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


def write_file_radar(file) -> None:
    path = f'../results'
    file_path = f"{path}"
    make_dir(file_path)
    file_name = f"{file_path}/{file.split('/')[-2]}_files.txt"
    with open(file_name, 'a') as txt_file:
        txt_file.write(f"{file}\n")
        txt_file.close()


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


def main():
    radar_name = "Carimagua"
    v = 2
    con = False if v == 3 else True
    zarr_store = f'/media/alfonso/drive/Alfonso/python/zarr_radar/{radar_name}_{v}.zarr'
    year, months, days = 2022, range(8, 9), range(9, 13)
    # year, months, days = 2022, range(3, 4), range(3, 4)
    for month in months:                                                                                                                                                                                                                                                                                                              
        for day in days:
            date_query = datetime(year=year, month=month, day=day)
            query = create_query(date=date_query, radar_site=radar_name)
            str_bucket = 's3://s3-radaresideam/'
            fs = fsspec.filesystem("s3", anon=True)
            radar_files = sorted(fs.glob(f"{str_bucket}{query}*"))
            if radar_files:
                start_time = time.monotonic()
                for i in radar_files:
                    exist = check_if_exist(i)
                    if not exist:
                        raw2dt(i, store=zarr_store, mode='a', consolidated=con, append_dim='vcp_time', zarr_version=v,
                               # elevation=[0.5]
                               )
                print(f"Run time for single{time.monotonic() - start_time} seconds")
                print('Done!!!')
            else:
                print(f'mes {month}, dia {day} no tienen datos')
        print(f'mes {month}, dia {day}')
    print('termine')
    pass


if __name__ == "__main__":
    main()
