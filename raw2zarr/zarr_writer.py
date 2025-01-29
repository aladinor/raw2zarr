from collections.abc import Mapping, MutableMapping
from os import PathLike
from typing import Any, Literal

from xarray import DataTree
from xarray.core.types import ZarrWriteModes


def dtree2zarr(
    dt: DataTree,
    store: MutableMapping | str | PathLike[str],
    mode: ZarrWriteModes = "w-",
    encoding: Mapping[str, Any] | None = None,
    consolidated: bool = True,
    group: str | None = None,
    write_inherited_coords: bool = False,
    compute: Literal[True] = True,
    **kwargs,
):
    """This function creates an appropriate datastore for writing a datatree
    to a zarr store.

    See `DataTree.to_zarr` for full API docs.
    """

    from zarr import consolidate_metadata

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if not compute:
        raise NotImplementedError("compute=False has not been implemented yet")

    if encoding is None:
        encoding = {}

    # In the future, we may want to expand this check to insure all the provided encoding
    # options are valid. For now, this simply checks that all provided encoding keys are
    # groups in the datatree.
    if set(encoding) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dt.groups)}"
        )
    append_dim = kwargs.pop("append_dim", None)
    for node in dt.subtree:
        at_root = node is dt
        if node.is_empty | node.is_root:
            continue
        ds = node.to_dataset(inherit=write_inherited_coords or at_root)
        group_path = None if at_root else "/" + node.relative_to(dt)
        try:
            ds.to_zarr(
                store,
                group=group_path,
                mode=mode,
                consolidated=False,
                append_dim=append_dim,
                **kwargs,
            )
        except ValueError as e:
            ds.to_zarr(
                store,
                group=group_path,
                mode="a-",
                encoding=encoding.get(node.path),
                consolidated=False,
                **kwargs,
            )

        if "w" in mode:
            mode = "a"

    if consolidated:
        consolidate_metadata(store)
