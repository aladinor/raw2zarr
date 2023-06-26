#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fsspec
import xarray as xr
import xradar as xd
import pandas as pd
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from functools import reduce
import numpy as np
from pandas import to_datetime
import matplotlib.pyplot as plt
from xmovie import Movie
from datetime import datetime
from datatree import DataTree, open_datatree
from utils import get_pars_from_ini

# radar_info = get_pars_from_ini('radar')
# print(1)


def create_query(date, radar_site):
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


def data_accessor(file):
    """
    Open AWS S3 file(s), which can be resolved locally by file caching
    """
    return fsspec.open_local(f'simplecache::s3://{file}', s3={'anon': True}, filecache={'cache_storage': '/tmp/radar/'})


def create_vcp(ls_dt):
    """
    Creates a tree-like object for each volume scan
    """
    dtree = [{f"{i[j].sweep_fixed_angle.values: .1f}".replace(' ', ''):
                  fix_angle(i[j].copy()).ds.xradar.georeference() for j in i.children} for i in ls_dt]
    dtree = reduce(lambda a, b: dict(a, **b), dtree)
    return DataTree.from_dict(dtree)


def create_path(dt):
    el = np.array([i for i in list(dt.children)])
    dt = pd.to_datetime(dt[f'{el[0]}'].time.values[0])
    return f"{dt:%Y%m%d%H%M}", dt


def rename_children(dt):
    for i in list(dt.children):
        dt[i].name = f"{dt[i].sweep_fixed_angle.values: .1f}".replace(' ', '')
    return dt


def new_structure(files):
    radar_name = files[0].split('/')[-1].split('.')[0][:3]
    elev = np.array(get_pars_from_ini('radar')[radar_name]['elevations'])
    swps = {j: f"sweep_{idx}" for idx, j in enumerate(elev)}
    data = {}
    for i in files:
        dt = xd.io.open_iris_datatree(data_accessor(i))
        data.update({float(dt[j].sweep_fixed_angle.values): fix_angle(dt[j]).ds.xradar.georeference()
                     for j in list(dt.children)})
    return DataTree.from_dict({swps[k]: data[k] for k in list(data.keys())})


def concat_dt(dt, times):
    dates = list(dt.keys())
    swps = list(dt[dates[0]].children)
    new_dt = {}
    for i in swps:
        ds = xr.concat([dt[j][i].ds for j in dates], dim='times')
        ds = ds.assign_coords(times=("times", times))
        new_dt.update({i: ds})
    return new_dt


def radar_dt(radar_files):
    paths = []
    ls_time = []
    dt_dict = {}
    for idx, i in enumerate(radar_files):
        if idx % 4 == 0:
            ls_files = radar_files[idx: idx + 4]
            vcp = new_structure(ls_files)
            path, time = create_path(vcp)
            paths.append(path)
            ls_time.append(time)
            dt_dict[f"{path}"] = vcp
    ds = concat_dt(dt_dict, ls_time)
    root_ds = xr.Dataset(
        {
            "vcp_time": xr.DataArray(
                paths,
                dims=["time"],
                coords={"time": pd.Series(list(ls_time)).sort_values()},
            ),
        },
        coords={
            "time": pd.Series(list(ls_time)).sort_values(),
        },
    )
    root_ds = root_ds.sortby("time")
    ds["/"] = root_ds
    return DataTree.from_dict(ds)


def mult_vcp(radar_files):
    """
    Creates a tree-like object for multiple volumes scan every 4th file in the bucket
    """
    paths = []
    ls_time = []
    dt_dict = {}
    for idx, i in enumerate(radar_files):
        if idx % 4 == 0:
            ls_files = radar_files[idx: idx + 4]
            # ls_sigmet = [xd.io.open_iris_datatree(data_accessor(j)).xradar.georeference() for j in ls_files]
            ls_sigmet = [xd.io.open_iris_datatree(data_accessor(j)) for j in ls_files]

            ls_sigmet = [rename_children(i) for i in ls_sigmet]
            vcp = create_vcp(ls_sigmet)
            path, time = create_path(vcp)
            paths.append(path)
            ls_time.append(time)
            dt_dict[f"{path}"] = vcp

    root_ds = xr.Dataset(
        {
            "vcp_time": xr.DataArray(
                paths,
                dims=["time"],
                coords={"time": pd.Series(list(ls_time)).sort_values()},
            ),
        },
        coords={
            "time": pd.Series(list(ls_time)).sort_values(),
        },
    )
    root_ds = root_ds.sortby("time")
    dt_dict["/"] = root_ds
    return DataTree.from_dict(dt_dict)


def fix_angle(ds):
    angle_dict = xd.util.extract_angle_parameters(ds)
    # display(angle_dict)
    start_ang = angle_dict["start_angle"]
    stop_ang = angle_dict["stop_angle"]
    angle_res = angle_dict["angle_res"]
    direction = angle_dict["direction"]

    # first find exact duplicates and remove
    ds = xd.util.remove_duplicate_rays(ds)

    # second reindex according to retrieved parameters
    ds = xd.util.reindex_angle(
        ds, start_ang, stop_ang, angle_res, direction, method="nearest"
    )
    return ds


def sel_by_date(dt, start, end=None):
    if end is None:
        paths = dt.vcp_time.sel(time=start).values
        times = dt.time.sel(time=start).values
    else:
        paths = dt.vcp_time.sel(time=slice(start, end)).values
        times = dt.time.sel(time=slice(start, end)).values

    ls_ds = [dt[i] for i in paths]
    dt_dict = {k: v for k, v in zip(paths, ls_ds)}
    root_ds = xr.Dataset(
        {
            "vcp_time": xr.DataArray(
                paths,
                dims=["time"],
                coords={"time": pd.Series(list(times)).sort_values()},
            ),
        },
        coords={
            "time": pd.Series(list(times)).sort_values(),
        },
    )
    root_ds = root_ds.sortby("time")
    dt_dict["/"] = root_ds
    return DataTree.from_dict(dt_dict)


def sel_by_elev(dt, elevation=1.3):
    ls_ds = []
    t = dt.time.values
    for i in dt['vcp_time'].values:
        try:
            data = dt[i][f"{elevation}"].ds
            ls_ds.append(data)
        except KeyError:
            print(f"please use the following elevation angles {list(dt[i].children)}")
    ds = xr.concat(ls_ds, dim=f"times", coords='all')
    ds.coords['times'] = ('times', t)
    return ds


def plt_ppi(ds):
    fig, ax = plt.subplots()
    ds.sel(times='2023-04-07 03:20', method='nearest').DBZH.plot(x="x", y='y', vmin=-10, vmax=50,
                                                                 cmap="Spectral_r", )
    m2km = lambda x, _: f"{x / 1000:g}"
    # set new ticks
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)
    ax.set_ylabel("$North - South \ distance \ [km]$")
    ax.set_xlabel("$East - West \ distance \ [km]$")
    ax.set_title(
        f"$Guaviare \ radar$"
        + "\n"
        + f"${to_datetime(ds.sel(times='2023-04-07 03:20', method='nearest').times.values): %Y-%m-%d - %H:%M}$"
        + "$ UTC$"
    )
    ax.set_ylim(-300000, 300000)
    ax.set_xlim(-300000, 300000)
    plt.show()


def plot_anim(ds):
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    proj_crs = xd.georeference.get_crs(ds)
    cart_crs = ccrs.Projection(proj_crs)
    sc = ds.isel(times=0).DBZH.plot.pcolormesh(x="x", y="y", vmin=-10,
                                               vmax=50, cmap="Spectral_r",
                                               transform=cart_crs,
                                               ax=ax)

    title = f"Barranca radar - {ds.isel(times=0).sweep_fixed_angle.values: .1f} [deg] \n " \
            f"{pd.to_datetime(ds.isel(times=0).time.values[0]):%Y-%m-%d %H:%M}UTC"
    ax.set_title(title)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color="gray", alpha=0.3, linestyle="--")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    gl.top_labels = False
    gl.right_labels = False
    ax.coastlines()

    def update_plot(t):
        sc.set_array(ds.sel(times=t).DBZH.values.ravel())
        ax.set_title(f"Barranca radar - {ds.sel(times=t).sweep_fixed_angle.values: .1f} [deg] \n "
                     f"{pd.to_datetime(ds.sel(times=t).time.values[0]):%Y-%m-%d %H:%M}UTC")

        ani = FuncAnimation(fig, update_plot, frames=ds.times.values,
                            interval=5)
        plt.show()
        # HTML(ani.to_html5_video())
        # writervideo = ani.FFMpegWriter(fps=60)
        # ani.save(f'/media/alfonso/drive/Alfonso/zarr_radar/ani.mp4')
        print(1)
        pass


def main():
    zarr_store = '../zarr/multiple_vcp_test.zarr'
    date_query = datetime(2023, 4, 7)
    radar_name = "Barrancabermeja"
    query = create_query(date=date_query, radar_site=radar_name)
    str_bucket = 's3://s3-radaresideam/'
    fs = fsspec.filesystem("s3", anon=True)

    date_query = datetime(2022, 8, 9, 14)
    radar_name = "Carimagua"
    query = create_query(date=date_query, radar_site=radar_name)
    str_bucket = 's3://s3-radaresideam/'
    fs = fsspec.filesystem("s3", anon=True)

    radar_files = sorted(fs.glob(f"{str_bucket}{query}*"))
    vcps_dt = radar_dt(radar_files[:24])
    _ = vcps_dt.to_zarr(zarr_store, mode='w', consolidated=True)
    del vcps_dt
    print('Done!!!')
    dt = open_datatree(zarr_store, engine="zarr")
    print(1)
    # # ds_le = sel_by_date(dt, start='2023-04-07 00:15', end='2023-04-07 00:55')
    # ds_le = sel_by_elev(dt, elevation=1.3)
    # mov = Movie
    # # plt_ppi(ds_le)
    # plot_anim(ds_le)
    # # _ = vcps_dt.to_zarr(zarr_store, mode='w')
    # ds = open_datatree(zarr_store, engine="zarr")
    # ds_surp = get_data(ds)
    print(1)
    pass


if __name__ == "__main__":
    main()
