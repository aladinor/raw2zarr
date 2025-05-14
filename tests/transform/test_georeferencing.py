import pytest
from xarray import DataTree

from raw2zarr.io.load import load_radar_data
from raw2zarr.transform.georeferencing import apply_georeferencing


@pytest.mark.slow
def test_apply_georeferencing(nexrad_aws_file_sweep10_misaligned):
    dtree = load_radar_data(nexrad_aws_file_sweep10_misaligned, engine="nexradlevel2")
    dtree_geo = apply_georeferencing(dtree)
    assert isinstance(dtree_geo, DataTree)
    for node in dtree_geo.match("sweep_*").groups:
        if node == "/":
            continue
        assert "x" in dtree_geo[node].ds.coords
        assert "y" in dtree_geo[node].ds.coords
        assert "x" in dtree_geo[node].ds.coords
