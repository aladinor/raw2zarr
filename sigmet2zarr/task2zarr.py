import time
import zarr
import xradar as xd
import numpy as np
from datatree import DataTree
from .utils import (
    get_pars_from_ini,
    data_accessor,
    fix_angle,
    convert_time,
    write_file_radar,
    load_toml,
)


def raw_to_dt(file: str) -> DataTree:
    """
    Function that convert sigmet files into a datatree using xd.io.open_iris_datatree
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
    return DataTree.from_dict({swps[k]: data[k] for k in list(data.keys())})


def dt2zarr2(dt: DataTree, **kwargs: dict) -> None:
    """
    Functions to save xradar datatree using zarr format
    @param dt: xradar datatree
    @param kwargs: keyword arguments
            - append_dim = dimension name where data will be appended. e.g. 'vcp_time'
            - mode = Xarray.to_zarr mode. Options {"w", "w-", "a", "a-", r+", None}
    @return: None
    """
    if kwargs["zarr_version"] == 3:
        st = zarr.DirectoryStoreV3(kwargs["store"])
    else:
        st = zarr.DirectoryStore(kwargs["store"])
    nodes = st.listdir()
    args = kwargs.copy()
    for child in list(dt.children):
        ds = dt[child].to_dataset()
        _time = convert_time(ds)
        ds[kwargs["append_dim"]] = _time
        ds = ds.expand_dims(dim=kwargs["append_dim"], axis=0).set_coords(
            kwargs["append_dim"]
        )
        if child in nodes:
            ds.to_zarr(group=child, **args)
        else:
            try:
                args = kwargs.copy()
                del args["append_dim"]
                args["mode"] = "w-"
                encoding = {
                    kwargs["append_dim"]: {
                        "units": "milliseconds since 2000-01-01T00:00:04.010000",
                        "dtype": "float",
                    },
                    "time": {
                        "units": "milliseconds since 2000-01-01T00:00:04.010000",
                        "dtype": "float",
                    },
                }
                ds.to_zarr(group=child, encoding=encoding, **args)
            except zarr.errors.ContainsGroupError:
                args = kwargs.copy()
                ds.to_zarr(group=child, **args)


def raw2zarr(file, **kwargs) -> None:
    """
    Main function to convert sigmet radar files into xradar datatree
    @param file: radar file path
    @param kwargs: keyword arguments passed to raw_to_dt
                - elevation: sweep elevation to be considered.e.g [0.5, 1.0, 3]. If None all sweeps will be considered
    @return: None
    """
    dt = raw_to_dt(file)
    elevations = [
        np.round(np.median(dt.children[i].elevation.data), 1) for i in list(dt.children)
    ]
    try:
        if kwargs["elevation"] in elevations:
            del kwargs["elevation"]
            dt2zarr2(dt=dt, **kwargs)
            del dt
            write_file_radar(file)
    except KeyError:
        dt2zarr2(dt=dt, **kwargs)
