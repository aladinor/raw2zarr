import pytest
import xarray as xr

from raw2zarr.builder.dtree_radar import radar_datatree


@pytest.fixture(scope="session")
def simple_nexrad_file(nexrad_local_gz_file):
    return nexrad_local_gz_file


@pytest.fixture(scope="session")
def dynamic_nexrad_file(nexrad_aws_file_SAILS):
    return nexrad_aws_file_SAILS


def test_simple_radar_datatree(simple_nexrad_file):
    """Test that radar_datatree returns a DataTree object."""
    append_dim = "vcp_time"
    radar_dtree = radar_datatree(
        simple_nexrad_file,
        engine="nexradlevel2",
        append_dim=append_dim,
    )
    assert isinstance(radar_dtree, xr.DataTree), "Expected an xarray.DataTree"
    ngroups = 12
    vcp_time = 1

    assert len(radar_dtree.groups) == ngroups, "Expected groups missing."
    for group in radar_dtree.groups:
        if group == "/":
            continue
        assert (
            append_dim in radar_dtree[group].coords
        ), f"Expected {append_dim} coordinate missing in {group}."
        assert (
            len(radar_dtree[group][append_dim]) == vcp_time
        ), f"Expected {vcp_time} values in {group}."


@pytest.mark.skip(
    reason="Skipping temporarily as dynamic scan aligment is not working properly yet"
)
def test_dynamic_radar_datatree(dynamic_nexrad_file):
    """Test that radar_datatree returns a DataTree object."""
    append_dim = "vcp_time"
    radar_dtree = radar_datatree(
        dynamic_nexrad_file,
        engine="nexradlevel2",
        append_dim=append_dim,
    )
    assert isinstance(radar_dtree, xr.DataTree), "Expected an xarray.DataTree"
    ngroups = 22
    vcp_time = 3

    assert len(radar_dtree.groups) == ngroups, "Expected groups missing."
    for group in radar_dtree.groups:
        if group == "/":
            continue
        assert (
            append_dim in radar_dtree[group].coords
        ), f"Expected {append_dim} coordinate missing in {group}."
        assert (
            len(radar_dtree[group][append_dim]) == vcp_time
        ), f"Expected {vcp_time} values in {group}."
