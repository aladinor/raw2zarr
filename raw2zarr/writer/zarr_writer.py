from collections.abc import Mapping, MutableMapping
from os import PathLike
from typing import Any, Literal

import icechunk
import xarray as xr
from xarray import DataTree
from xarray.core.types import ZarrWriteModes

from ..builder.dtree_radar import radar_datatree
from ..templates.template_utils import remove_string_vars
from ..transform.encoding import dtree_encoding


def dtree_to_zarr(
    dtree: DataTree,
    store: MutableMapping | str | PathLike[str],
    mode: ZarrWriteModes = "w-",
    encoding: Mapping[str, Any] | None = None,
    consolidated: bool = False,
    group: str | None = None,
    write_inherited_coords: bool = False,
    compute: Literal[True] = True,
    zarr_format: int = 3,
    region: dict[str, slice] | None = None,
    append_dim: str | None = None,
    **kwargs,
) -> None:
    """This function creates an appropriate datastore for writing a datatree
    to a zarr store.

    See `DataTree.to_zarr` for full API docs.
    """

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if encoding is None:
        encoding = {}

    if set(encoding) - set(dtree.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dtree.groups)}"
        )
    for node in dtree.subtree:

        at_root = node is dtree
        if node.is_empty | node.is_root:
            continue

        ds = node.to_dataset(inherit=write_inherited_coords or at_root)
        group_path = None if at_root else "/" + node.relative_to(dtree)
        ds.to_zarr(
            store,
            group=group_path,
            mode=mode,
            encoding=encoding.get(group_path, {}),
            zarr_format=zarr_format,
            consolidated=consolidated,
            compute=compute,
            append_dim=append_dim,
            region=region,
            **kwargs,
        )


def drop_vars_region(dtree: xr.DataTree, append_dim: str) -> xr.DataTree:
    """
    Drop variables that don't have the append dimension for region writing.

    Args:
        dtree: DataTree to process
        append_dim: Dimension name for appending data

    Returns:
        DataTree with variables filtered for region writing
    """

    def drop_vars_no_append_dim(ds: xr.Dataset, append_dim: str) -> xr.Dataset:
        drop_list = [var for var in ds.variables if append_dim not in ds[var].dims]
        return ds.drop_vars(drop_list)

    return dtree.map_over_datasets(drop_vars_no_append_dim, append_dim)


def write_dtree_region(
    file: str,
    idx: int,
    session: icechunk.Session,
    append_dim: str,
    engine: str,
    zarr_format: int = 3,
    consolidated: bool = False,
    remove_strings=True,
    **kwargs,
) -> icechunk.Session:
    """
    Write radar data to a specific region in an existing Zarr store.

    Args:
        file: Path to radar file to process
        idx: Index position for writing in the append dimension
        session: Icechunk session for store access
        append_dim: Dimension name for appending data
        engine: Engine for reading radar files
        zarr_format: Zarr format version
        consolidated: Whether to consolidate metadata
        remove_strings: Whether to remove string variables
        **kwargs: Additional arguments passed to dtree_to_zarr

    Returns:
        Updated icechunk session
    """
    dtree = radar_datatree(file, engine=engine)
    # TODO: remove this after strings are supported by zarr v3
    if remove_strings:
        dtree = remove_string_vars(dtree)
        dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)

    region = {append_dim: slice(idx, idx + 1)}

    writer_args = dict(
        store=session.store,
        mode="a-",
        zarr_format=zarr_format,
        consolidated=consolidated,
        write_inherited_coords=True,
        region=region,
        **kwargs,
    )
    dtree_append = drop_vars_region(dtree, append_dim=append_dim)
    dtree_to_zarr(dtree_append, **writer_args)
    return session
