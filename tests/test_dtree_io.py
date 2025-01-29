import pytest
from xarray import DataTree

from raw2zarr.dtree_io import nexradlevel2_loader


@pytest.mark.parametrize(
    "nexradlevel2_files",
    ["nexrad_s3_file", "nexrad_local_str_file", "nexrad_gz_local_file"],
    indirect=True,
)
def test_nexrad_loader(nexradlevel2_files):
    """
    Test nexradlevel2_loader with different file formats (GZIP and BZ2).
    Ensures the output is a valid DataTree and contains expected structure.
    """
    dtree = nexradlevel2_loader(nexradlevel2_files)
    expected_groups = 11
    expected_sweeps = 7
    # Assertions with better error messages
    assert isinstance(
        dtree, DataTree
    ), f"Expected dtree to be of type DataTree, but got {type(dtree)}"
    assert len(dtree.groups) == expected_groups, (
        f"Expected {expected_groups} groups, but found {len(dtree.groups)}. "
        f"Check the input file: {nexradlevel2_files}"
    )
    assert len(dtree.match("sweep*")) == expected_sweeps, (
        f"Expected {expected_sweeps} sweeps, but found {len(dtree.match('sweep*'))}. "
        f"Possible issue with sweep extraction in: {nexradlevel2_files}"
    )
    assert (
        "DBZH" in dtree["sweep_0"].data_vars
    ), f"'DBZH' variable missing in 'sweep_0'. Available data_vars: {list(dtree['sweep_0'].data_vars.keys())}"
    assert (
        "azimuth" in dtree["sweep_0"].coords
    ), f"'azimuth' coordinate missing in 'sweep_0'. Available coords: {list(dtree['sweep_0'].coords.keys())}"
