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
    test_config = load_json_config("vcp_nexrad.json")
    assert vcp in config, f"{vcp} not found in loaded config"

    vcp_spec = test_config["VCP-11"]

    # Ensure required keys exist for unified config
    assert "elevations" in vcp_spec
    assert "dims" in vcp_spec

    # Check that we have sweep configurations
    sweep_count = 0
    for key in vcp_spec.keys():
        if key.startswith("sweep_"):
            sweep_count += 1

    assert sweep_count > 0, "No sweep configurations found"
    assert sweep_count == len(
        vcp_spec["elevations"]
    ), "Mismatch in elevation count vs sweep count"

    assert all(
        isinstance(e, (float, int)) for e in vcp_spec["elevations"]
    ), "Non-numeric elevation found"

    assert (
        vcp_spec["elevations"] == config["VCP-11"]["elevations"]
    ), "Elevations do not match"
