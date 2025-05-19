import numpy as np
from packaging.version import parse as parse_version

from raw2zarr.transform.encoding import dtree_encoding


def test_dtree_encoding_full_structure(radar_dtree_factory):
    append_dim = "vcp_time"
    dtree = radar_dtree_factory(append_dim="vcp_time")
    encoding = dtree_encoding(dtree, append_dim)

    assert "/" in encoding
    root_enc = encoding["/"]
    assert append_dim in root_enc
    assert root_enc[append_dim]["units"].startswith("nanoseconds")
    assert root_enc[append_dim]["_FillValue"] == -9999

    if parse_version(np.__version__) >= parse_version("2.0.0"):
        str_encoding = np.dtypes.StringDType
    else:
        str_encoding = np.dtype("U")

    for field in [
        "platform_type",
        "instrument_type",
        "time_coverage_start",
        "time_coverage_end",
    ]:
        assert root_enc[field]["dtype"] == str_encoding

    assert "/sweep_0" in encoding
    sweep_enc = encoding["/sweep_0"]

    assert "time" in sweep_enc
    assert sweep_enc["time"]["units"].startswith("nanoseconds")
    assert append_dim in sweep_enc
    assert sweep_enc[append_dim]["units"].startswith("nanoseconds")

    for field in ["DBZH", "ZDR", "PHIDP", "RHOHV"]:
        assert sweep_enc[field]["_FillValue"] == -9999

    for field in ["sweep_mode", "prt_mode", "follow_mode"]:
        assert sweep_enc[field]["dtype"] == str_encoding

    for field in ["sweep_number", "sweep_fixed_angle"]:
        assert sweep_enc[field]["_FillValue"] == -9999
