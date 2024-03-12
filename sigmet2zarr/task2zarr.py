import os
import shutil
import zarr
import xradar as xd
import numpy as np
from datatree import DataTree
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
    elev = np.array(load_toml("../config/radar.toml")[radar_name]["elevations"])
    swps = {j: f"sweep_{idx}" for idx, j in enumerate(elev)}
    data = {}
    dt = xd.io.open_iris_datatree(data_accessor(file))
    data.update(
        {
            float(dt[j].sweep_fixed_angle.values): fix_angle(
                dt[j]
            ).ds.xradar.georeference()
            for j in list(dt.children)
        }
    )
    shutil.rmtree(cache_storage)
    return DataTree.from_dict({swps[k]: data[k] for k in list(data.keys())})


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
    st = (
        zarr.DirectoryStoreV3(zarr_store)
        if zarr_version == 3
        else zarr.DirectoryStore(zarr_store)
    )
    nodes = st.listdir()
    for child in list(dt.children):
        ds = dt[child].to_dataset()
        _time = convert_time(ds)
        ds[append_dim] = _time
        ds = ds.expand_dims(dim=append_dim, axis=0).set_coords(append_dim)
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
            encoding = {
                append_dim: {
                    "units": "milliseconds since 2000-01-01T00:00:04.010000",
                    "dtype": "float",
                },
                "time": {
                    "units": "milliseconds since 2000-01-01T00:00:04.010000",
                    "dtype": "float",
                },
            }
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
) -> None:
    """
    Main function to convert sigmet radar files into xradar datatree and save them using zarr format
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
    dt = raw_to_dt(file)
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
