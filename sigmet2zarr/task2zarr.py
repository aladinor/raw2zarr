import os
import shutil
import datatree
import zarr
import xradar as xd
import numpy as np
from datatree import DataTree
from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from xarray import full_like
from sigmet2zarr.utils import (
    data_accessor,
    fix_angle,
    convert_time,
    write_file_radar,
    load_toml,
    time_encoding,
)


def _get_root(dt: DataTree):
    groups = [
        i for i in list(dt.groups) if not i.startswith("/sweep") if i not in ["/"]
    ]
    root = DataTree(data=dt.root.ds, name="root")
    for group in groups:
        DataTree(data=dt[group].ds, name=group[1:], parent=root)
    return root


def _fix_sn(dt: DataTree, sw_num: list[int]):
    groups = [i for i in list(dt.groups) if i.startswith("/sweep")]
    for group in groups:
        sn: float = float(dt[group].ds.sweep_fixed_angle.values)
        nsn: int = sw_num[sn]
        new_sn = full_like(dt[group].ds.sweep_number, nsn)
        dt[group]["sweep_number"] = new_sn
    return dt


def raw_to_dt(
    file: str, append_dim: str, cache_storage: str = "/tmp/radar/"
) -> DataTree:
    """
    Function that convert sigmet files into a datatree using xd.io.open_iris_datatree
    @param cache_storage: locally caching remote files path
    @param file: radar file path
    @return: xradar datatree with all sweeps within each file
    """
    radar_name = file.split("/")[-1].split(".")[0][:3]
    elev: np.array = np.array(
        load_toml("../config/radar.toml")[radar_name]["elevations"]
    )
    sw_num: np.array = np.array(
        load_toml("../config/radar.toml")[radar_name]["sweep_number"]
    )
    swps: dict[float, str] = {j: f"sweep_{idx}" for idx, j in enumerate(elev)}
    sw_fix: dict[float, int] = {j: sw_num[idx] for idx, j in enumerate(elev)}
    data: dict[float, Dataset] = {}
    dt: DataTree = xd.io.open_iris_datatree(data_accessor(file, cache_storage))
    _fix_sn(dt, sw_fix)
    data.update(
        {
            float(dt[j].sweep_fixed_angle.values): fix_angle(
                dt[j]
            ).ds.xradar.georeference()
            for j in list(dt.children)
            if j not in ["radar_parameters"]
        }
    )
    data = exp_dim(data, append_dim=append_dim)
    dtree = _get_root(dt)
    for i, sw in enumerate(data.keys()):
        DataTree(data[sw], name=swps[sw], parent=dtree)
    shutil.rmtree(cache_storage, ignore_errors=True)
    return dtree


def exp_dim(dt, append_dim) -> DataTree:
    """
    Functions that expand dimension to each dataset within the datatree
    @param dt: xarray.datatree
    @param append_dim: dimension name which dataset will be expanded. e.g. 'vcp_time'
    @return: xarray Datatree
    """
    dt_new = {}
    for sw, ds in dt.items():
        _time = convert_time(ds)
        if not _time:
            continue
        ds[append_dim] = _time
        ds: Dataset = ds.expand_dims(dim=append_dim, axis=0).set_coords(append_dim)
        dt_new[sw] = ds
    return dt_new


def dt2zarr2(
    dt: DataTree,
    zarr_store: str,
    zarr_version: int,
    append_dim: str,
    mode: str,
    consolidated: bool,
) -> None:
    """
    Functions to save xradar datatree using zarr format
    @param consolidated: Xarray consolidated metadata. Default True
    @param append_dim: dimension name where data will be appended. e.g. 'vcp_time'
    @param mode: Xarray.to_zarr mode. Options are "w", "w-", "a", "a-", r+", None
    @param zarr_version: data can be store in zarr format using version 2 or 3. Default V=2
    @param zarr_store: path to zarr store
    @param dt: xradar datatree
    @return: None
    """
    st: zarr.DirectoryStore = (
        zarr.DirectoryStoreV3(zarr_store)
        if zarr_version == 3
        else zarr.DirectoryStore(zarr_store)
    )
    nodes = st.listdir()
    if not nodes:
        encoding: dict = time_encoding(dt, append_dim)
        dt.to_zarr(
            mode=mode,
            store=zarr_store,
            zarr_version=zarr_version,
            consolidated=consolidated,
            encoding=encoding,
        )
    else:
        children = [i for i in list(dt.children) if i.startswith("sweep")]
        for child in children:
            ds = dt[child].to_dataset()
            encoding = time_encoding(ds, append_dim)
            if child in nodes:
                ds.to_zarr(
                    group=child,
                    mode=mode,
                    store=zarr_store,
                    zarr_version=zarr_version,
                    consolidated=consolidated,
                    append_dim=append_dim,
                )
            else:
                mode = "w-"
                ds.to_zarr(
                    group=child,
                    mode=mode,
                    store=zarr_store,
                    zarr_version=zarr_version,
                    consolidated=consolidated,
                    encoding=encoding,
                )


def raw2zarr(
    file: str,
    zarr_store: str,
    zarr_version: int = 2,
    elevation: list[float] = None,
    append_dim: str = "vcp_time",
    mode: str = "a",
    consolidated: bool = True,
    p2c: str = "../results",
    cache_storage: str = "/tmp/radar/",
) -> None:
    """
    Main function to convert sigmet radar files into xradar datatree and save them using zarr format
    @param cache_storage: locally caching remote files path
    @param consolidated: Xarray consolidated metadata. Default True
    @param p2c: path to write a file where each radar filename will be written once is processed.
    @param mode:  Xarray.to_zarr mode. Options are "w", "w-", "a", "a-", r+", None
    @param append_dim: dimension name where data will be appended. e.g. 'vcp_time'
    @param elevation: list of elevation to be converted into zarr.
                      E.g. [0.5, 1.0, 3]. If None all sweeps within the radar object will be considered
    @param zarr_version: data can be store in zarr format using version 2 or 3. Default V=2
    @param zarr_store: path to zarr store
    @param file: radar file path
    @return: None
    """
    dt = raw_to_dt(file, append_dim=append_dim, cache_storage=cache_storage)
    elevations = [
        np.round(np.median(dt.children[i].elevation.data), 1)
        for i in list(dt.children)
        if i not in ["radar_parameters"]
    ]
    if not elevation:
        dt2zarr2(
            dt=dt,
            zarr_store=zarr_store,
            zarr_version=zarr_version,
            mode=mode,
            consolidated=consolidated,
            append_dim=append_dim,
        )
        write_file_radar(file, p2c)
    elif elevation in elevations:
        dt2zarr2(
            dt=dt,
            zarr_store=zarr_store,
            zarr_version=zarr_version,
            mode=mode,
            consolidated=consolidated,
            append_dim=append_dim,
        )
        write_file_radar(file, p2c)
