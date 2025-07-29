import os
import shutil

import numpy as np
import pytest
import xarray as xr

from raw2zarr.builder.builder_utils import get_icechunk_repo
from raw2zarr.builder.executor import append_parallel, append_sequential
from tests.builder.conftest import requires_numpy2_and_zarr3


@pytest.fixture(scope="session")
def sample_nexrad_file(nexrad_local_gz_file):
    return nexrad_local_gz_file


@pytest.fixture(scope="session")
def sample_nexrad_files(nexrad_local_gz_files):
    return nexrad_local_gz_files


@pytest.fixture(scope="session")
def output_zarr(tmp_path_factory):
    path = tmp_path_factory.mktemp("zarr_test") / "output.zarr"
    yield str(path)
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def test_cluster():
    """Create a mock cluster that runs synchronously for fast tests."""
    from unittest.mock import MagicMock
    import dask

    # Mock cluster that looks real but uses synchronous scheduler
    mock_cluster = MagicMock()
    mock_client = MagicMock()

    # Mock scheduler_info to return fake workers
    mock_client.scheduler_info.return_value = {
        "workers": {"worker-1": {}, "worker-2": {}}
    }

    # Override map/gather to run synchronously for speed
    def sync_map(func, iterable):
        """Run map synchronously instead of distributed."""
        with dask.config.set(scheduler="synchronous"):
            return [func(item) for item in iterable]

    def sync_gather(results):
        """Return results immediately."""
        return results

    mock_client.map = sync_map
    mock_client.gather = sync_gather

    # Mock the cluster's scheduler_info
    mock_cluster.scheduler_info = mock_client.scheduler_info

    # Patch the Client constructor to return our mock
    import dask.distributed

    original_client = dask.distributed.Client
    dask.distributed.Client = lambda cluster, **kwargs: mock_client

    yield mock_cluster

    # Restore original Client
    dask.distributed.Client = original_client


@requires_numpy2_and_zarr3
def test_append_sequential_creates_zarr(sample_nexrad_files, output_zarr):
    append_dim = "vcp_time"
    repo = get_icechunk_repo(output_zarr)
    append_sequential(
        radar_files=sample_nexrad_files,
        repo=repo,
        append_dim=append_dim,
        engine="nexradlevel2",
    )

    assert os.path.exists(output_zarr), "Expected Zarr store not found."
    session = repo.readonly_session("main")
    ngroups = 11
    vcp_time = 2
    radar_dtree = xr.open_datatree(
        session.store,
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


@requires_numpy2_and_zarr3
@pytest.mark.serial
def test_append_parallel_creates_zarr(sample_nexrad_files, output_zarr, test_cluster):
    append_dim = "vcp_time"
    repo = get_icechunk_repo(output_zarr)
    append_parallel(
        radar_files=sample_nexrad_files,
        repo=repo,
        append_dim=append_dim,
        engine="nexradlevel2",
        cluster=test_cluster,
    )

    assert os.path.exists(output_zarr), "Expected Zarr store not found."

    ngroups = 11
    vcp_time = 2
    session = repo.readonly_session("main")
    radar_dtree = xr.open_datatree(
        session.store,
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


@requires_numpy2_and_zarr3
@pytest.mark.serial
def test_parallel_vs_sequential_equivalence(sample_nexrad_file, tmp_path, test_cluster):
    zarr_seq = tmp_path / "zarr_seq.zarr"
    zarr_par = tmp_path / "zarr_par.zarr"
    repo_seq = get_icechunk_repo(zarr_seq)
    append_sequential(
        radar_files=[sample_nexrad_file],
        repo=repo_seq,
        append_dim="vcp_time",
        engine="nexradlevel2",
        remove_strings=True,
    )
    repo_par = get_icechunk_repo(zarr_par)
    append_parallel(
        repo=repo_par,
        radar_files=[sample_nexrad_file],
        append_dim="vcp_time",
        engine="nexradlevel2",
        remove_strings=True,
        cluster=test_cluster,
    )
    session_seq = repo_seq.readonly_session("main")
    tree_seq = xr.open_datatree(
        session_seq.store,
        engine="zarr",
        consolidated=False,
        chunks={},
        zarr_format=3,
    )
    session_par = repo_par.readonly_session("main")
    tree_par = xr.open_datatree(
        session_par.store,
        engine="zarr",
        consolidated=False,
        chunks={},
        zarr_format=3,
    )

    assert sorted(tree_seq.groups) == sorted(
        tree_par.groups
    ), "Mismatch in group structure"

    for group in tree_seq.groups:
        ds_seq = tree_seq[group].ds  # .compute()
        ds_par = tree_par[group].ds  # .compute()

        assert ds_seq.dims == ds_par.dims, f"Dimension mismatch in {group}"
        assert set(ds_seq.data_vars) == set(
            ds_par.data_vars
        ), f"Data variables mismatch in {group}"

        for var in ds_seq.data_vars:
            # Compare data values only, not coordinate metadata/chunking
            seq_values = ds_seq[var].values
            par_values = ds_par[var].values
            np.testing.assert_allclose(
                seq_values,
                par_values,
                rtol=1e-5,
                err_msg=f"Data values differ for variable {var} in {group}",
            )
