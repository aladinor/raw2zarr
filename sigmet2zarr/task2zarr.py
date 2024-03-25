import os
import shutil
import datatree
import zarr
import xradar as xd
import numpy as np
from datatree import DataTree
from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from xarray import concat
from sigmet2zarr.utils import (
    data_accessor,
    fix_angle,
    convert_time,
    write_file_radar,
    load_toml,
)


def raw_to_dt(file: str, cache_storage: str = "/tmp/radar/") -> DataTree:
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
    data: dict[float, Dataset] = {}
    dt: DataTree = xd.io.open_iris_datatree(data_accessor(file, cache_storage))
    data.update(
        {
            float(dt[j].sweep_fixed_angle.values): fix_angle(
                dt[j]
            ).ds.xradar.georeference()
            for j in list(dt.children)
            if j not in ["radar_parameters"]
        }
    )
    sn = {elev[i]: sw_num[i] for i in range(len(elev))}
    act_sn = np.array(
        [
            sn[float(dt[j].sweep_fixed_angle.values)]
            for j in list(dt.children)
            if j not in ["radar_parameters"]
        ]
    )
    dtree: DataTree = DataTree(data=dt.root.ds, name="root")
    dtree["sweep_number"] = DataArray(act_sn, dims={"sweep": np.array(len(act_sn))})
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
    data_new: dict = {}
    for child in list(dt.children):
        ds: Dataset = dt[child].to_dataset()
        _time = convert_time(ds)
        if not _time:
            continue
        ds[append_dim] = _time
        ds: Dataset = ds.expand_dims(dim=append_dim, axis=0).set_coords(append_dim)
        data_new[child] = ds
    dtree: DataTree = DataTree(name="root", data=dt.root.ds)
    for sw in data_new.keys():
        DataTree(data_new[sw], name=sw, parent=dtree)
    return dtree


def time_encoding(dtree, append_dim) -> dict:
    encoding = {}
    if type(dtree) is DataTree:
        for group in list(dtree.groups)[1:]:
            encoding.update(
                {
                    f"{group}": {
                        f"{append_dim}": dict(
                            units="nanoseconds since 2000-01-01T00:00:00.00",
                            dtype="float64",
                            _FillValue=np.datetime64("NaT"),
                        ),
                        "time": dict(
                            units="nanoseconds since 2000-01-01T00:00:00.00",
                            dtype="float64",
                            _FillValue=np.datetime64("NaT"),
                        ),
                    }
                }
            )
        return encoding
    else:
        encoding.update(
            {
                f"{append_dim}": dict(
                    units="nanoseconds since 2000-01-01T00:00:00.00",
                    dtype="float64",
                    _FillValue=np.datetime64("NaT"),
                ),
                "time": dict(
                    units="nanoseconds since 2000-01-01T00:00:00.00",
                    dtype="float64",
                    _FillValue=np.datetime64("NaT"),
                ),
            }
        )
        return encoding


def update_sweep_num(st, dtree):
    dt = datatree.open_datatree(st, engine="zarr")
    root = dt.root.to_dataset()
    mr = concat(
        [root["sweep_number"], dtree.root.ds["sweep_number"]], join="inner", dim="sweep"
    )
    dt_root = dt.root.ds.drop_dims("sweep").assign(sweep_number=mr)
    groups = [x for xs in [sw.children for sw in [dt, dtree]] for x in xs]
    dtr = DataTree(dt_root, name="root")
    for group in [dt.children, dtree.children]:
        DataTree(data=dt[group].ds, name=group, parent=dtr)
    dtr.to_zarr(
        mode="w",
        store=st,
    )


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
    # todo: check the root dataset and add the sweep_number and sweep_fixed angle before writing
    dtree = exp_dim(dt, append_dim=append_dim)
    if not nodes:
        encoding: dict = time_encoding(dtree, append_dim)
        dtree.to_zarr(
            mode=mode,
            store=zarr_store,
            zarr_version=zarr_version,
            consolidated=consolidated,
            encoding=encoding,
        )
    else:
        # update_sweep_num(st, dtree)
        for child in list(dtree.children):
            ds = dtree[child].to_dataset()
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
    dt = raw_to_dt(file, cache_storage=cache_storage)
    elevations = [
        np.round(np.median(dt.children[i].elevation.data), 1) for i in list(dt.children)
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
