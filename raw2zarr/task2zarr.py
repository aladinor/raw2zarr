import zarr
import xradar as xd
import numpy as np
from xarray import full_like, Dataset, DataTree
from zarr.errors import ContainsGroupError
from raw2zarr.utils import (
    data_accessor,
    fix_angle,
    write_file_radar,
    load_toml,
    dtree_encoding,
    exp_dim,
)


def _get_root(dt: DataTree):
    groups = [
        i for i in list(dt.groups) if not i.startswith("/sweep") if i not in ["/"]
    ]
    root = DataTree(data=dt.root.ds, name="root")
    for group in groups:
        DataTree(data=dt[group].ds, name=group[1:], parent=root)
    return root


def _fix_sn(ds: Dataset, sw_num: dict[float, int]) -> dict:
    sn: float = float(ds["sweep_fixed_angle"].values)
    nsn: int = sw_num[sn]
    new_sn = full_like(ds.sweep_number, nsn)
    new_ds = ds.copy(deep=True)
    new_ds["sweep_number"] = new_sn
    return new_ds


def prepare2append(dt: DataTree, append_dim: str, radar_name: str = "GUA") -> DataTree:
    """
    Converts SIGMET radar files into a DataTree structure and prepares it for appending along a specified dimension.

    This function processes a given DataTree of radar data, organizes it by sweep angles, and prepares it for appending
    along the specified dimension. It uses configuration files to map radar sweep angles and numbers, and georeferences
    the data before appending.

    Parameters
    ----------
    dt : DataTree
        The DataTree object containing radar data to be processed.
    append_dim : str
        The dimension along which the data will be appended (e.g., time, elevation).
    radar_name : str, optional
        The radar name to identify the correct configuration (default is "GUA").

    Returns
    -------
    DataTree
        A new DataTree object with all sweeps processed and ready for appending along the specified dimension.

    Notes
    -----
    - The function expects a configuration file in TOML format located at "../config/radar.toml", containing
      the necessary radar sweep angle and sweep number information.
    - Each sweep in the DataTree is georeferenced, and its sweep number is corrected before being organized
      into the final DataTree structure.

    Examples
    --------
    >>> radar_data = prepare2append(my_datatree, append_dim="time", radar_name="GUA")
    >>> # radar_data is now prepared for appending along the time dimension
    """
    elev: np.array = np.array(
        load_toml("../config/radar.toml")[radar_name]["elevations"]
    )
    sw_num: np.array = np.array(
        load_toml("../config/radar.toml")[radar_name]["sweep_number"]
    )
    swps: dict[float, str] = {j: f"sweep_{idx}" for idx, j in enumerate(elev)}
    sw_fix: dict[float, int] = {j: sw_num[idx] for idx, j in enumerate(elev)}

    tree = {
        node.path: node.to_dataset()
        for node in dt.subtree
        if not node.path.startswith("/sweep")
    }
    tree.update(
        {
            swps[float(node.sweep_fixed_angle.values)]: fix_angle(
                _fix_sn(node, sw_num=sw_fix)
            )
            .to_dataset()
            .xradar.georeference()
            for node in dt.subtree
            if node.path.startswith("/sweep")
        }
    )
    tree = exp_dim(tree, append_dim=append_dim)
    return DataTree.from_dict(tree)


def dt2zarr2(
    dt: DataTree,
    zarr_store: str,
    zarr_version: int,
    append_dim: str,
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

    for node in dt.subtree:
        ds = node.to_dataset()
        group_path = node.path
        if group_path.startswith("/sweep"):
            encoding = dtree_encoding(ds, append_dim)
        else:
            encoding = {}
        try:
            ds.to_zarr(
                store=st,
                group=group_path,
                mode="w-",
                encoding=encoding,
                consolidated=consolidated,
            )
        except ContainsGroupError:
            try:
                ds.to_zarr(
                    store=st,
                    group=group_path,
                    mode="a-",
                    consolidated=consolidated,
                    append_dim="vcp_time",
                )
            except ValueError:
                continue


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
    dt: DataTree = xd.io.open_iris_datatree(data_accessor(file))
    dtree = prepare2append(dt, append_dim=append_dim)
    elevations = [
        np.round(np.median(dtree.children[i].elevation.data), 1)
        for i in list(dtree.children)
        if i not in ["radar_parameters"]
    ]
    if not elevation:
        dt2zarr2(
            dt=dtree,
            zarr_store=zarr_store,
            zarr_version=zarr_version,
            mode=mode,
            consolidated=consolidated,
            append_dim=append_dim,
        )
        write_file_radar(file, p2c)
    elif elevation in elevations:
        dt2zarr2(
            dt=dtree,
            zarr_store=zarr_store,
            zarr_version=zarr_version,
            mode=mode,
            consolidated=consolidated,
            append_dim=append_dim,
        )
        write_file_radar(file, p2c)
