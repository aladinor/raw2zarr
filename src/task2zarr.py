#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import fsspec
import time
import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
import zarr
from datatree import DataTree
from utils import get_pars_from_ini


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
    return fsspec.open_local(f'simplecache::s3://{file}', s3={'anon': True}, filecache={'cache_storage': '/tmp/radar/'})


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


def dt2zarr2(dt, store, **kwargs) -> None:
    st = zarr.DirectoryStore(store)
    nodes = st.listdir()
    args = kwargs.copy()
    for child in list(dt.children):
        ds = dt[child].to_dataset()
        _time = get_time(ds)
        ds['times'] = _time
        ds = ds.expand_dims(dim='times', axis=0).set_coords('times')
        if child in nodes:
            try:
                ds.to_zarr(store=st, group=child, **args)
            except ValueError as e:
                print(e)
                print('el error es aca')
                pass
        else:
            try:
                args = kwargs.copy()
                del args['append_dim']
                args['mode'] = 'w-'
                encoding = {'times': {'units': 'nanoseconds since 1970-01-01', 'dtype': 'int64'}}
                ds.to_zarr(store=st, group=child, encoding=encoding, **args)
            except zarr.errors.ContainsGroupError:
                args = kwargs.copy()
                ds.to_zarr(store=st, group=child, **args)


def raw2dt(files, store, mode, consolidated, append_dim) -> None:
    dt = raw_to_dt(files)
    dt2zarr2(dt=dt, store=store, mode=mode, consolidated=consolidated, append_dim=append_dim)


def fix_angle(ds) -> xr.Dataset:
    angle_dict = xd.util.extract_angle_parameters(ds)
    # display(angle_dict)
    start_ang = angle_dict["start_angle"]
    stop_ang = angle_dict["stop_angle"]
    angle_res = angle_dict["angle_res"]
    if angle_res > 1:
        angle_res = np.round(angle_res, 1)
    direction = angle_dict["direction"]
    tol = angle_res / 1.75
    # first find exact duplicates and remove
    ds = xd.util.remove_duplicate_rays(ds)

    # second reindex according to retrieved parameters
    ds = xd.util.reindex_angle(
        ds, start_ang, stop_ang, angle_res, direction, method="nearest",
        tolerance=tol
    )
    return ds


def main():

    zarr_store = '/media/alfonso/drive/Alfonso/zarr_radar/bag.zarr'
    os.system(f"rm -rf {zarr_store}")
    for j in [6, 7, 8]:
        date_query = datetime(2023, 4, j)
        radar_name = "Barrancabermeja"
        query = create_query(date=date_query, radar_site=radar_name)
        str_bucket = 's3://s3-radaresideam/'
        fs = fsspec.filesystem("s3", anon=True)
        radar_files = sorted(fs.glob(f"{str_bucket}{query}*"))
        start_time = time.monotonic()
        for i in radar_files:
            raw2dt(i, store=zarr_store, mode='a', consolidated=True, append_dim='times')
        print(f"Run time for single{time.monotonic() - start_time} seconds")
        print('Done!!!')
    pass


if __name__ == "__main__":
    main()
