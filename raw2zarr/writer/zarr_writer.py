import time
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
    store: MutableMapping | str | PathLike[str] | None = None,
    mode: ZarrWriteModes = "w-",
    encoding: Mapping[str, Any] | None = None,
    consolidated: bool = False,
    group: str | None = None,
    write_inherited_coords: bool = False,
    compute: Literal[True] = True,
    zarr_format: int = 3,
    region: dict[str, slice] | None = None,
    append_dim: str | None = None,
    use_icechunk: bool | None = None,
    session: icechunk.Session | None = None,
    **kwargs,
) -> None:
    """Write DataTree to Zarr store with automatic backend selection.

    This function intelligently selects between xarray's to_zarr() and icechunk's
    to_icechunk() based on the store type and write mode:

    - Template creation (compute=False): Uses to_zarr()
    - Region writes to Icechunk store: Uses to_icechunk() for proper session tracking
    - All other cases: Uses to_zarr()

    Args:
        dtree: DataTree to write
        store: Zarr store or path (mutually exclusive with session)
        mode: Write mode ('w-', 'a', 'a-', etc.)
        encoding: Encoding specifications per group
        consolidated: Whether to consolidate metadata
        group: Root group (not yet implemented)
        write_inherited_coords: Whether to write inherited coordinates
        compute: Whether to compute immediately (False for templates)
        zarr_format: Zarr format version (2 or 3)
        region: Region specification for parallel writes
        append_dim: Dimension name for appending
        use_icechunk: Force icechunk backend (None=auto-detect from session)
        session: Icechunk session (mutually exclusive with store)
        **kwargs: Additional arguments passed to to_zarr

    Raises:
        ValueError: If both or neither of store/session provided
        ValueError: If use_icechunk=True but session not provided
        NotImplementedError: If group parameter is specified

    See Also:
        https://icechunk.io/en/latest/parallel/
        DataTree.to_zarr documentation
    """

    # Validate mutually exclusive parameters
    if (store is None) == (session is None):
        raise ValueError("Exactly one of 'store' or 'session' must be provided")

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    # Auto-detect icechunk usage from session parameter if not specified
    if use_icechunk is None:
        use_icechunk = session is not None

    # Validate icechunk requirements
    if use_icechunk and session is None:
        raise ValueError(
            "session parameter is required when use_icechunk=True. "
            "Pass an icechunk.Session object or set use_icechunk=False"
        )

    # Determine which store to use
    write_store = session.store if session is not None else store

    # Set up encoding
    if encoding is None:
        encoding = {}

    # Validate encoding group names
    unexpected_groups = set(encoding) - set(dtree.groups)
    if unexpected_groups:
        raise ValueError(
            f"unexpected encoding group name(s) provided: {unexpected_groups}"
        )

    # Choose write path
    if use_icechunk and region is not None:
        _write_with_icechunk(dtree, session, region, write_inherited_coords, compute)
    else:
        _write_with_zarr(
            dtree,
            write_store,
            mode,
            encoding,
            zarr_format,
            consolidated,
            write_inherited_coords,
            compute,
            append_dim,
            region,
            **kwargs,
        )


def _write_with_icechunk(
    dtree: DataTree,
    session: icechunk.Session,
    region: dict[str, slice],
    write_inherited_coords: bool,
    compute: bool,
) -> None:
    """Write DataTree using icechunk backend for parallel region writes.

    Args:
        dtree: DataTree to write
        session: Icechunk session for store access
        region: Region specification for parallel writes
        write_inherited_coords: Whether to write inherited coordinates
        compute: Whether to compute dask arrays before writing

    Note:
        to_icechunk() doesn't accept compute, encoding, zarr_format, consolidated,
        or mode parameters. These are handled automatically by icechunk.
    """
    from icechunk.xarray import to_icechunk

    start_total = time.time()
    node_count = 0
    total_compute_time = 0
    total_write_time = 0

    for node in dtree.subtree:
        at_root = node is dtree

        # Skip empty and root nodes
        if node.is_empty or node.is_root:
            continue

        node_count += 1

        # Convert node to dataset
        ds = node.to_dataset(inherit=write_inherited_coords or at_root)

        # Materialize dask arrays before writing
        # Note: to_icechunk() doesn't accept a compute parameter,
        # so we must materialize arrays explicitly when compute=True
        compute_start = time.time()
        if compute:
            ds = ds.compute()
        compute_elapsed = time.time() - compute_start
        total_compute_time += compute_elapsed

        # Build group path: None for root node, "/path" for children
        if at_root:
            group_path = None
        else:
            group_path = "/" + node.relative_to(dtree)

        # Write using icechunk backend
        # Note: to_icechunk() automatically handles zarr_format,
        # consolidated metadata, and mode settings
        write_start = time.time()
        to_icechunk(
            ds,
            session=session,
            group=group_path,
            region=region,
        )
        write_elapsed = time.time() - write_start
        total_write_time += write_elapsed

        print(
            f"[to_icechunk] group={group_path or 'root'}: compute={compute_elapsed:.4f}s, write={write_elapsed:.4f}s"
        )

    total_elapsed = time.time() - start_total
    overhead = total_elapsed - total_compute_time - total_write_time
    print(
        f"[to_icechunk] TOTAL: {node_count} nodes in {total_elapsed:.4f}s (compute={total_compute_time:.4f}s, write={total_write_time:.4f}s, overhead={overhead:.4f}s)"
    )


def _write_with_zarr(
    dtree: DataTree,
    store: MutableMapping | str | PathLike[str],
    mode: str,
    encoding: dict,
    zarr_format: int,
    consolidated: bool,
    write_inherited_coords: bool,
    compute: bool,
    append_dim: str | None,
    region: dict[str, slice] | None,
    **kwargs,
) -> None:
    """Write DataTree using standard xarray to_zarr() backend.

    Args:
        dtree: DataTree to write
        store: Zarr store or path
        mode: Write mode ('w-', 'a', 'a-', etc.)
        encoding: Encoding specifications per group
        zarr_format: Zarr format version (2 or 3)
        consolidated: Whether to consolidate metadata
        write_inherited_coords: Whether to write inherited coordinates
        compute: Whether to compute immediately (False for templates)
        append_dim: Dimension name for appending
        region: Region specification for writes
        **kwargs: Additional arguments passed to to_zarr
    """
    start_total = time.time()
    node_count = 0

    for node in dtree.subtree:
        at_root = node is dtree

        # Skip empty and root nodes
        if node.is_empty or node.is_root:
            continue

        node_count += 1

        # Convert node to dataset
        ds = node.to_dataset(inherit=write_inherited_coords or at_root)

        # Build group path: None for root node, "/path" for children
        if at_root:
            group_path = None
        else:
            group_path = "/" + node.relative_to(dtree)

        # Write using standard zarr backend
        write_start = time.time()
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
        write_elapsed = time.time() - write_start

        print(f"[to_zarr] group={group_path or 'root'}: write={write_elapsed:.4f}s")

    total_elapsed = time.time() - start_total
    print(f"[to_zarr] TOTAL: {node_count} nodes in {total_elapsed:.4f}s")


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
    remove_strings: bool = True,
    is_dynamic: bool = False,
    sweep_indices: list[int] | None = None,
    elevation_angles: list[float] | None = None,
    vcp_config_file: str = "vcp_nexrad.json",
    **kwargs,
) -> icechunk.Session:
    """Write radar data to a specific region in an existing Zarr store.

    This function always uses the icechunk backend (use_icechunk=True)
    since it requires session-based region writes for parallel processing.

    Args:
        file: Path to radar file to process
        idx: Index position for writing in the append dimension
        session: Icechunk session for store access
        append_dim: Dimension name for appending data
        engine: Engine for reading radar files
        zarr_format: Zarr format version (passed through, may be ignored by icechunk)
        consolidated: Whether to consolidate metadata (passed through, may be ignored)
        remove_strings: Whether to remove string variables
        is_dynamic: Whether to use template-based processing for dynamic scans
        sweep_indices: Sweep indices to include (for temporal slicing)
        elevation_angles: Elevation angles for this temporal slice (for VCP sweep mapping)
        vcp_config_file: VCP configuration file name
        **kwargs: Additional arguments passed to dtree_to_zarr

    Returns:
        Updated icechunk session

    Note:
        zarr_format and consolidated parameters are passed to dtree_to_zarr()
        but may be ignored by the to_icechunk() backend, which handles these
        settings automatically.
    """
    dtree = radar_datatree(
        file,
        engine=engine,
        append_dim=append_dim,
        is_dynamic=is_dynamic,
        sweep_indices=sweep_indices,
        elevation_angles=elevation_angles,
        vcp_config_file=vcp_config_file,
    )
    # TODO: remove this after strings are supported by zarr v3
    if remove_strings:
        dtree = remove_string_vars(dtree)
        dtree.encoding = dtree_encoding(dtree, append_dim=append_dim)

    # TODO: all backends should return sweep_group_name and sweep_fixed_angle at the root. (Missing in NExrad and Iris)
    remove_root_vars = True
    if remove_root_vars:
        root_path = list(dtree.children)[0]
        dtree_dict = dtree.to_dict()
        try:
            dtree_dict[f"/{root_path}"] = dtree_dict[f"/{root_path}"].drop_vars(
                ["sweep_group_name", "sweep_fixed_angle"]
            )
            dtree = DataTree.from_dict(dtree_dict)
        except ValueError:
            dtree = DataTree.from_dict(dtree_dict)

    region = {append_dim: slice(idx, idx + 1)}

    writer_args = dict(
        session=session,
        mode="a-",
        zarr_format=zarr_format,
        consolidated=consolidated,
        write_inherited_coords=True,
        region=region,
        compute=True,
        use_icechunk=True,
        **kwargs,
    )
    dtree_append = drop_vars_region(dtree, append_dim=append_dim)
    dtree_to_zarr(dtree_append, **writer_args)
    return session
