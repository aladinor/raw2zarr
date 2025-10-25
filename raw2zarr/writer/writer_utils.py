from typing import Any

import xarray as xr
from icechunk import Repository, Session
from icechunk.session import Session as IcechunkSession
from xarray import DataTree, open_datatree

from ..builder.dtree_radar import radar_datatree
from ..templates.template_utils import remove_string_vars
from ..templates.vcp_utils import create_multi_vcp_template
from ..transform.encoding import dtree_encoding
from .zarr_writer import dtree_to_zarr


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
        enc = None if has_dim else encoding
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


def init_zarr_store(
    files: list[tuple[int, str]],
    session: IcechunkSession,
    append_dim: str,
    engine: str,
    zarr_format: int,
    consolidated: bool,
    remove_strings: bool = True,
    vcp_time_mapping: dict | None = None,
    vcp_config_file: str = "vcp_nexrad.json",
) -> list[tuple[int, str]]:
    """
    Initialize Zarr store with VCP-specific templates if it doesn't exist.

    Args:
        files: List of (index, filepath) tuples
        session: Icechunk session for store access
        append_dim: Dimension name for appending data
        engine: Engine for reading radar files
        zarr_format: Zarr format version
        consolidated: Whether to consolidate metadata
        remove_strings: Whether to remove string variables
        vcp_time_mapping: VCP time mapping for multi-VCP support
        vcp_config_file: VCP configuration file name in the config directory

    Returns:
        List of remaining files to process
    """
    exis_zarr_store = zarr_store_has_append_dim(
        session.store,
        append_dim=append_dim,
    )
    if not exis_zarr_store:
        idx, first_file = files.pop(0)
        dtree = radar_datatree(first_file, engine=engine)
        try:
            vcp = dtree[dtree.groups[1]].attrs["scan_name"].strip()
        except KeyError:
            vcp = "DEFAULT"
        radar_info = {
            "lon": dtree[vcp].longitude.item(),
            "lat": dtree[vcp].latitude.item(),
            "alt": dtree[vcp].altitude.item(),
            "crs_wkt": dtree[f"{vcp}/sweep_0"].ds["crs_wkt"].attrs,
            "reference_time": dtree[vcp].time_coverage_start.item(),
            "vcp": vcp,
            "instrument_name": dtree[vcp].attrs["instrument_name"],
            "volume_number": dtree[vcp].volume_number.item(),
            "platform_type": dtree[vcp].platform_type.item(),
            "instrument_type": dtree[vcp].instrument_type.item(),
            "time_coverage_start": dtree[vcp].time_coverage_start.item(),
            "time_coverage_end": dtree[vcp].time_coverage_end.item(),
        }

        # Create individual VCP templates and combine them
        final_tree = create_multi_vcp_template(
            vcp_time_mapping=vcp_time_mapping,
            base_radar_info=radar_info,
            append_dim=append_dim,
            remove_strings=remove_strings,
            vcp_config_file=vcp_config_file,
        )
        if remove_strings:
            final_tree = remove_string_vars(final_tree)
            final_tree.encoding = dtree_encoding(final_tree, append_dim=append_dim)

        # Write combined template to store
        writer_args = resolve_zarr_write_options(
            store=session.store,
            group_path=None,
            encoding=final_tree.encoding,
            append_dim=append_dim,
            zarr_format=zarr_format,
            consolidated=consolidated,
            compute=False,
        )
        dtree_to_zarr(final_tree, **writer_args)

        session.commit("Initial commit: VCP-specific xarray template created")
        # Return all files including the first one for region writing
        files.insert(0, (idx, first_file))
        return files
    return files


def check_cords(repo: Repository) -> None:
    required_cords = ["x", "y", "z", "time"]

    session = repo.readonly_session("main")
    dtree = open_datatree(
        session.store,
        zarr_format=3,
        consolidated=False,
        chunks=None,
        engine="zarr",
    )

    def cords_in_sweeps(dt: DataTree):
        """Check if all sweeps have required coordinates"""
        for path in dt.match("*/sweep_*").groups[2:]:
            ds = dt[path].ds
            has_coords = all(coord in ds.coords for coord in required_cords)
            if not has_coords:
                missing = [c for c in required_cords if c not in ds.coords]
                print(f"Missing coordinates in {path}: {missing}")
                return False
        return True

    cords_exist = cords_in_sweeps(dtree)

    if not cords_exist:
        print("Fixing coordinate variables...")
        # Try coordinate-only write first (efficient)
        try:
            session = repo.writable_session("main")
            print("Attempting coordinate-only fix (fast)...")
            for path in dtree.match("*/sweep_*").groups[2:]:
                ds = dtree[path].ds
                coords_to_set = [
                    coord
                    for coord in required_cords
                    if coord in ds.data_vars and coord not in ds.coords
                ]

                if coords_to_set:
                    # Create minimal dataset with just the coordinates to fix
                    coord_data = {coord: ds[coord] for coord in coords_to_set}
                    coord_ds = xr.Dataset(coord_data).set_coords(coords_to_set)

                    # Write only the coordinate metadata updates
                    coord_ds.to_zarr(
                        session.store,
                        group=path,
                        mode="a",
                        consolidated=False,
                        zarr_format=3,
                    )

            snapshot_id = session.commit("Fixed coordinate variables")
            print(f"Coordinate fix committed as snapshot {snapshot_id}")

        except Exception as e:
            print(f"Coordinate-only fix failed: {e}")

    else:
        print("All coordinates are properly set.")
