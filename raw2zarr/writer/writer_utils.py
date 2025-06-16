from typing import Any

import xarray as xr
from icechunk import Session


def zarr_store_has_append_dim(
    store: Session.store, append_dim: str, group_path: str | None = None
) -> bool:
    try:
        dt = xr.open_datatree(
            store,
            consolidated=False,
            zarr_format=3,
            chunks=None,
            engine="zarr",
            group=group_path,
        )
        exist_dim = []
        for node in dt.subtree:
            dims = node.ds.dims
            exist_dim.extend(dims)
        if append_dim in set(exist_dim):
            return True
        return False
    except FileNotFoundError:
        return False


def resolve_zarr_write_options(
    store,
    group_path: str | None,
    append_dim: str | None,
    encoding: dict[str, Any] | None = None,
    region: dict[str, slice] | None = None,
    consolidated: bool = False,
    default_mode: str = "w-",
    zarr_format: int = 3,
    write_inherited_coords: bool = True,
    compute: bool = True,
):
    """Determine correct mode, append_dim, and encoding for a Zarr write."""
    is_region_write = region is not None
    has_dim = zarr_store_has_append_dim(
        store=store,
        group_path=group_path,
        append_dim=append_dim,
    )

    if is_region_write:
        app_dim = None
        mode = "a"
        enc = None
    else:
        app_dim = append_dim if has_dim else None
        enc = (
            None
            if has_dim
            else (encoding if group_path in ["/", None] else encoding[group_path])
        )
        mode = "a-" if has_dim else default_mode

    return {
        "store": store,
        "mode": mode,
        "encoding": enc,
        "consolidated": consolidated,
        "zarr_format": zarr_format,
        "append_dim": app_dim,
        "region": region,
        "write_inherited_coords": write_inherited_coords,
        "compute": compute,
    }


def drop_vars_region(dtree: xr.DataTree, append_dim: str) -> xr.DataTree:
    def drop_vars_no_append_dim(ds: xr.Dataset, append_dim: str) -> xr.Dataset:
        drop_list = [var for var in ds.variables if append_dim not in ds[var].dims]
        return ds.drop_vars(drop_list)

    return dtree.map_over_datasets(drop_vars_no_append_dim, append_dim)
