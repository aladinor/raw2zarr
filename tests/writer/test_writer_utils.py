from __future__ import annotations

import icechunk
import numpy as np
import pytest
import xarray as xr
from xarray import DataTree, open_datatree

from raw2zarr.writer.writer_utils import check_cords

SWEEP_PATH = "VCP-12/sweep_0"
REQUIRED_COORDS: tuple[str, ...] = ("x", "y", "z", "time")


@pytest.fixture()
def repo() -> icechunk.Repository:
    storage = icechunk.in_memory_storage()
    return icechunk.Repository.create(storage)


@pytest.fixture()
def radar_dtree_with_coord_issues() -> DataTree:
    """Radar-style DataTree where x,y,z,time are data variables (not coords)."""
    n_gates = 100
    n_rays = 360

    x_data = np.random.randn(n_rays, n_gates) * 1000
    y_data = np.random.randn(n_rays, n_gates) * 1000
    z_data = np.random.randn(n_rays, n_gates) * 100
    time_data = np.array(
        [
            np.datetime64("2023-01-01T00:00:00") + np.timedelta64(i, "s")
            for i in range(n_rays)
        ]
    )

    azimuth = np.linspace(0, 360, n_rays, endpoint=False)
    range_data = np.arange(n_gates) * 250
    reflectivity = np.random.randn(n_rays, n_gates) * 10 + 20

    sweep_ds = xr.Dataset(
        {
            "x": (["azimuth", "range"], x_data),
            "y": (["azimuth", "range"], y_data),
            "z": (["azimuth", "range"], z_data),
            "time": (["azimuth"], time_data),
            "reflectivity": (["azimuth", "range"], reflectivity),
            "crs_wkt": ([], "PROJCS[...]"),
        },
        coords={"azimuth": azimuth, "range": range_data},
    )

    vcp_ds = xr.Dataset(
        {
            "longitude": ([], -95.0),
            "latitude": ([], 35.0),
            "altitude": ([], 200.0),
            "time_coverage_start": ([], "2023-01-01T00:00:00"),
            "time_coverage_end": ([], "2023-01-01T00:05:00"),
        }
    )
    vcp_ds.attrs.update(
        {
            "scan_name": "VCP-12",
            "instrument_name": "Test Radar",
            "volume_number": 1,
            "platform_type": "fixed",
            "instrument_type": "radar",
        }
    )

    root_ds = xr.Dataset()
    dtree_dict = {"/": root_ds, "/VCP-12": vcp_ds, f"/{SWEEP_PATH}": sweep_ds}
    return DataTree.from_dict(dtree_dict)


@pytest.fixture()
def radar_dtree_coords_fixed(radar_dtree_with_coord_issues: DataTree) -> DataTree:
    """Same as above, but with x,y,z,time promoted to coordinates."""
    sweep_ds = radar_dtree_with_coord_issues[SWEEP_PATH].ds
    corrected = sweep_ds.set_coords(list(REQUIRED_COORDS))
    dtree_dict = radar_dtree_with_coord_issues.to_dict()
    dtree_dict[f"/{SWEEP_PATH}"] = corrected
    return DataTree.from_dict(dtree_dict)


class TestCheckCords:
    def test_fixes_coordinate_variables(
        self, radar_dtree_with_coord_issues: DataTree, repo: icechunk.Repository
    ) -> None:
        from raw2zarr.writer.zarr_writer import dtree_to_zarr

        # Write dtree with coord issues
        ws = repo.writable_session("main")
        dtree_to_zarr(
            radar_dtree_with_coord_issues,
            store=ws.store,
            mode="w-",
            consolidated=False,
            zarr_format=3,
            write_inherited_coords=True,
        )
        ws.commit("initial data")

        # Assert initial state (x,y,z,time are data vars)
        rs = repo.readonly_session("main")
        dt = open_datatree(
            rs.store, zarr_format=3, consolidated=False, chunks=None, engine="zarr"
        )
        ds = dt[SWEEP_PATH].ds
        for c in REQUIRED_COORDS:
            assert c in ds.data_vars and c not in ds.coords

        check_cords(repo)

        rs = repo.readonly_session("main")
        dt = open_datatree(
            rs.store, zarr_format=3, consolidated=False, chunks=None, engine="zarr"
        )
        ds = dt[SWEEP_PATH].ds
        for c in REQUIRED_COORDS:
            assert c in ds.coords and c not in ds.data_vars

    def test_noop_when_coordinates_correct(
        self, radar_dtree_coords_fixed: DataTree, repo: icechunk.Repository
    ) -> None:
        from raw2zarr.writer.zarr_writer import dtree_to_zarr

        # Write dtree with correct coords
        ws = repo.writable_session("main")
        dtree_to_zarr(
            radar_dtree_coords_fixed,
            store=ws.store,
            mode="w-",
            consolidated=False,
            zarr_format=3,
            write_inherited_coords=True,
        )
        ws.commit("initial data")

        rs = repo.readonly_session("main")
        dt = open_datatree(
            rs.store, zarr_format=3, consolidated=False, chunks=None, engine="zarr"
        )
        ds = dt[SWEEP_PATH].ds
        for c in REQUIRED_COORDS:
            assert c in ds.coords and c not in ds.data_vars

        check_cords(repo)

        rs = repo.readonly_session("main")
        dt = open_datatree(
            rs.store, zarr_format=3, consolidated=False, chunks=None, engine="zarr"
        )
        ds = dt[SWEEP_PATH].ds
        for c in REQUIRED_COORDS:
            assert c in ds.coords and c not in ds.data_vars
