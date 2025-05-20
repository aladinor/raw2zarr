import os
import shutil

import pytest
import xarray as xr

from raw2zarr.builder.runner import append_parallel, append_sequential


@pytest.fixture(scope="session")
def sample_nexrad_file(nexrad_local_gz_file):
    return nexrad_local_gz_file


@pytest.fixture(scope="session")
def output_zarr(tmp_path_factory):
    path = tmp_path_factory.mktemp("zarr_test") / "output.zarr"
    yield str(path)
    if os.path.exists(path):
        shutil.rmtree(path)


def test_append_sequential_creates_zarr(sample_nexrad_file, output_zarr):
    append_dim = "vcp_time"
    append_sequential(
        radar_files=[sample_nexrad_file, sample_nexrad_file],
        append_dim=append_dim,
        zarr_store=output_zarr,
        engine="nexradlevel2",
    )

    assert os.path.exists(output_zarr), "Expected Zarr store not found."

    ngroups = 11
    vcp_time = 2
    radar_dtree = xr.open_datatree(
        output_zarr,
        engine="zarr",
        consolidated=False,
        chunks={},
        zarr_format=3,
    )
    assert len(radar_dtree.groups) == ngroups, "Expected Zarr groups missing."
    for group in radar_dtree.groups:
        if group == "/":
            continue
        assert (
            append_dim in radar_dtree[group].coords
        ), f"Expected {append_dim} coordinate missing in {group}."
        assert (
            len(radar_dtree[group][append_dim]) == vcp_time
        ), f"Expected {vcp_time} values in {group}."


def test_append_parallel_creates_zarr(sample_nexrad_file, output_zarr):
    append_dim = "vcp_time"
    append_parallel(
        radar_files=[sample_nexrad_file, sample_nexrad_file],
        append_dim=append_dim,
        zarr_store=output_zarr,
        engine="nexradlevel2",
    )

    assert os.path.exists(output_zarr), "Expected Zarr store not found."

    ngroups = 11
    vcp_time = 2
    radar_dtree = xr.open_datatree(
        output_zarr,
        engine="zarr",
        consolidated=False,
        chunks={},
        zarr_format=3,
    )

    assert len(radar_dtree.groups) == ngroups, "Expected Zarr groups missing."
    for group in radar_dtree.groups:
        if group == "/":
            continue
        assert (
            append_dim in radar_dtree[group].coords
        ), f"Missing coord {append_dim} in {group}"
        assert (
            len(radar_dtree[group][append_dim]) == vcp_time
        ), f"Expected {vcp_time} values in {group}"


def test_parallel_vs_sequential_equivalence(sample_nexrad_file, tmp_path):
    zarr_seq = tmp_path / "zarr_seq.zarr"
    zarr_par = tmp_path / "zarr_par.zarr"

    append_sequential(
        radar_files=[sample_nexrad_file],
        append_dim="vcp_time",
        zarr_store=str(zarr_seq),
        engine="nexradlevel2",
    )

    append_parallel(
        radar_files=[sample_nexrad_file],
        append_dim="vcp_time",
        zarr_store=str(zarr_par),
        engine="nexradlevel2",
    )

    tree_seq = xr.open_datatree(
        str(zarr_seq),
        engine="zarr",
        consolidated=False,
        chunks={},
        zarr_format=3,
    )

    tree_par = xr.open_datatree(
        str(zarr_par),
        engine="zarr",
        consolidated=False,
        chunks={},
        zarr_format=3,
    )

    assert sorted(tree_seq.groups) == sorted(
        tree_par.groups
    ), "Mismatch in group structure"

    for group in tree_seq.groups:
        ds_seq = tree_seq[group].ds
        ds_par = tree_par[group].ds

        assert ds_seq.dims == ds_par.dims, f"Dimension mismatch in {group}"
        assert set(ds_seq.data_vars) == set(
            ds_par.data_vars
        ), f"Data variables mismatch in {group}"

        for var in ds_seq.data_vars:
            xr.testing.assert_allclose(ds_seq[var], ds_par[var], rtol=1e-5)
