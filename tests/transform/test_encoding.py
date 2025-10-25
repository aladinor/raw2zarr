from raw2zarr.transform.encoding import dtree_encoding


class TestDTreeEncoding:
    def test_full_structure(self, radar_dtree_factory):
        append_dim = "vcp_time"
        dtree = radar_dtree_factory(append_dim="vcp_time")
        encoding = dtree_encoding(dtree, append_dim)

        assert "/" in encoding
        root_enc = encoding["/"]
        assert append_dim in root_enc
        assert root_enc[append_dim]["units"].startswith("nanoseconds")
        assert root_enc[append_dim]["_FillValue"] == -9999

        # Check that string fields have Unicode string dtypes
        # The actual dtype (U5, U7, U20, etc.) depends on the string length
        for field in [
            "platform_type",
            "instrument_type",
            "time_coverage_start",
            "time_coverage_end",
        ]:
            dtype = root_enc[field]["dtype"]
            # Accept either dtype object or string representation
            dtype_str = str(dtype) if hasattr(dtype, "str") else dtype
            assert dtype_str.startswith("U") or dtype_str.startswith(
                "<U"
            ), f"{field} dtype should be Unicode string, got {dtype}"

        assert "/sweep_0" in encoding
        sweep_enc = encoding["/sweep_0"]

        assert "time" in sweep_enc
        assert sweep_enc["time"]["units"].startswith("nanoseconds")
        assert append_dim in sweep_enc
        assert sweep_enc[append_dim]["units"].startswith("nanoseconds")

        for field in ["DBZH", "ZDR", "PHIDP", "RHOHV"]:
            assert sweep_enc[field]["_FillValue"] == -999.0

        # Check that sweep-level string fields have Unicode string dtypes
        for field in ["sweep_mode", "prt_mode", "follow_mode"]:
            dtype = sweep_enc[field]["dtype"]
            # Accept either dtype object or string representation
            dtype_str = str(dtype) if hasattr(dtype, "str") else dtype
            assert dtype_str.startswith("U") or dtype_str.startswith(
                "<U"
            ), f"{field} dtype should be Unicode string, got {dtype}"

        for field in ["sweep_number", "sweep_fixed_angle"]:
            assert sweep_enc[field]["_FillValue"] == -9999
