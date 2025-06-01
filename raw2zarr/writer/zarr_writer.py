from collections.abc import Mapping, MutableMapping
from os import PathLike
from typing import Any, Literal

from xarray import DataTree
from xarray.core.types import ZarrWriteModes


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
