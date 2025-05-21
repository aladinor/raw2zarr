import numpy as np
import pytest

from raw2zarr.io.load import load_radar_data
from raw2zarr.transform.alignment import check_dynamic_scan, fix_angle


@pytest.mark.slow
def test_fix_angle_on_misaligned_sweep10(nexrad_aws_file_sweep10_misaligned):
    dtree = load_radar_data(nexrad_aws_file_sweep10_misaligned, engine="nexradlevel2")
    sweep_number = "sweep_10"
    sweep = dtree[sweep_number].ds
    assert len(sweep.azimuth) < 360

    dtree_fixed = fix_angle(dtree)
    sweep_fixed = dtree_fixed[sweep_number].ds
    expected_azimuth = np.arange(0.5, 360.0, 1.0)

    assert len(sweep_fixed.azimuth) == 360
    assert np.allclose(np.diff(sweep_fixed.azimuth), 1.0)
    assert np.allclose(sweep_fixed.azimuth.values, expected_azimuth, atol=0.01)


def test_check_dynamic_scan(nexrad_aws_file_SAILS):
    dtree = load_radar_data(nexrad_aws_file_SAILS, engine="nexradlevel2")
    is_dynamic = check_dynamic_scan(dtree)
    assert is_dynamic is True


def test_check_dynamic_scan_detects_static(nexrad_local_uncompressed_file):
    dtree = load_radar_data(nexrad_local_uncompressed_file, engine="nexradlevel2")
    is_dynamic = check_dynamic_scan(dtree)
    assert is_dynamic is False
