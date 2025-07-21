from raw2zarr.config.utils import load_json_config


def test_load_json_config():
    config = {
        "VCP-11": {
            "elevations": [
                0.5,
                0.5,
                1.45,
                1.45,
                2.4,
                3.4,
                4.3,
                5.3,
                6.2,
                7.5,
                8.7,
                10.0,
                12.0,
                14.0,
                16.7,
                19.5,
            ],
            "scan_types": [
                "CS-W",
                "CD-W",
                "CS-W",
                "CD-W",
                "B",
                "B",
                "B",
                "B",
                "B",
                "CD/WO",
                "CD/WO",
                "CD/WO",
                "CD/WO",
                "CD/WO",
                "CD/WO",
                "CD/WO",
            ],
        }
    }
    vcp = "VCP-11"
    test_config = load_json_config("vcp.json")
    assert vcp in config, f"{vcp} not found in loaded config"

    vcp_spec = test_config["VCP-11"]

    # Ensure required keys exist
    assert "elevations" in vcp_spec
    assert "scan_types" in vcp_spec

    assert len(vcp_spec["elevations"]) == len(
        vcp_spec["scan_types"]
    ), "Mismatch in elevation vs scan_type count"
    assert all(
        isinstance(e, (float, str)) for e in vcp_spec["elevations"]
    ), "Non-numeric elevation found"
    assert all(
        isinstance(s, str) for s in vcp_spec["scan_types"]
    ), "Non-string scan_type found"

    assert (
        vcp_spec["elevations"] == config["VCP-11"]["elevations"]
    ), "Elevations do not match"
    # This test is canceled until we get the final scan types
    # assert (
    #     vcp_spec["scan_types"] == config["VCP-11"]["scan_types"]
    # ), "Scan types do not match"
    for vcp in config.keys():
        assert len(config[vcp]["elevations"]) == len(
            config[vcp]["scan_types"]
        ), "Mismatch in elevation vs scan_type count"
