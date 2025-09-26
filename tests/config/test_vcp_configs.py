"""
Tests for VCP configuration files validation and consistency.
"""

import json
from pathlib import Path

import pytest

from raw2zarr.templates.template_manager import VcpTemplateManager


class TestVcpConfigStructure:
    """Test VCP configuration file structure and consistency."""

    @pytest.fixture
    def config_dir(self):
        """Get the config directory path."""
        return Path(__file__).resolve().parent.parent.parent / "raw2zarr" / "config"

    def test_config_directory_exists(self, config_dir):
        """Test that config directory exists."""
        assert config_dir.exists(), f"Config directory not found: {config_dir}"
        assert config_dir.is_dir(), f"Config path is not a directory: {config_dir}"

    def test_nexrad_config_exists(self, config_dir):
        """Test that default NEXRAD config exists."""
        nexrad_config = config_dir / "vcp_nexrad.json"
        assert nexrad_config.exists(), "vcp_nexrad.json not found"

    def test_ideam_config_exists(self, config_dir):
        """Test that IDEAM config exists."""
        ideam_config = config_dir / "ideam.json"
        assert ideam_config.exists(), "ideam.json not found"

    def test_eccc_config_exists(self, config_dir):
        """Test that ECCC config exists."""
        eccc_config = config_dir / "eccc.json"
        assert eccc_config.exists(), "eccc.json not found"

    def test_config_files_are_valid_json(self, config_dir):
        """Test that all config files are valid JSON."""
        config_files = ["vcp_nexrad.json", "ideam.json", "eccc.json"]

        for config_file in config_files:
            config_path = config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {config_file}: {e}")


class TestIdeamConfigValidation:
    """Test IDEAM config file validation."""

    @pytest.fixture
    def ideam_config(self):
        """Load IDEAM config for testing."""
        try:
            manager = VcpTemplateManager("ideam.json")
            return manager.config
        except FileNotFoundError:
            pytest.skip("IDEAM config file not available")

    def test_ideam_required_vcps(self, ideam_config):
        """Test that IDEAM config has all required VCP patterns."""
        required_vcps = ["PRECC", "SURVP", "PRECA", "PRECB"]

        for vcp in required_vcps:
            assert vcp in ideam_config, f"Missing required VCP: {vcp}"

    def test_ideam_vcp_structure(self, ideam_config):
        """Test that IDEAM VCPs have correct structure."""
        for vcp_name, vcp_config in ideam_config.items():
            # Required top-level fields
            assert "elevations" in vcp_config, f"{vcp_name} missing elevations"
            assert "dims" in vcp_config, f"{vcp_name} missing dims"

            # Elevations should be list of numbers
            elevations = vcp_config["elevations"]
            assert isinstance(elevations, list), f"{vcp_name} elevations not a list"
            assert len(elevations) > 0, f"{vcp_name} has no elevations"

            for i, elev in enumerate(elevations):
                assert isinstance(
                    elev, (int, float)
                ), f"{vcp_name} elevation {i} not numeric"
                assert 0 <= elev <= 90, f"{vcp_name} elevation {i} out of range: {elev}"

    def test_ideam_sweep_configurations(self, ideam_config):
        """Test that IDEAM VCPs have sweep configurations."""
        for vcp_name, vcp_config in ideam_config.items():
            elevations = vcp_config["elevations"]

            # Check that sweep configurations exist for each elevation
            for i in range(len(elevations)):
                sweep_key = f"sweep_{i}"
                assert sweep_key in vcp_config, f"{vcp_name} missing {sweep_key}"

                sweep_config = vcp_config[sweep_key]
                assert (
                    "variables" in sweep_config
                ), f"{vcp_name} {sweep_key} missing variables"

                # Check variable structure
                variables = sweep_config["variables"]
                assert isinstance(
                    variables, dict
                ), f"{vcp_name} {sweep_key} variables not dict"
                assert len(variables) > 0, f"{vcp_name} {sweep_key} has no variables"

    def test_ideam_variable_structure(self, ideam_config):
        """Test that IDEAM variables have correct structure."""
        for vcp_name, vcp_config in ideam_config.items():
            for key, value in vcp_config.items():
                if key.startswith("sweep_") and isinstance(value, dict):
                    if "variables" in value:
                        for var_name, var_config in value["variables"].items():
                            # Required fields for each variable
                            required_fields = ["dtype", "fill_value", "attributes"]
                            for field in required_fields:
                                assert (
                                    field in var_config
                                ), f"{vcp_name} {key} {var_name} missing {field}"

                            # Attributes should be a dict
                            assert isinstance(
                                var_config["attributes"], dict
                            ), f"{vcp_name} {key} {var_name} attributes not dict"

    def test_ideam_specific_variables(self, ideam_config):
        """Test that IDEAM configs have expected radar variables."""
        # IDEAM should have DBZH (main reflectivity variable)
        dbzh_found = False

        for vcp_name, vcp_config in ideam_config.items():
            for key, value in vcp_config.items():
                if key.startswith("sweep_") and isinstance(value, dict):
                    if "variables" in value:
                        if "DBZH" in value["variables"]:
                            dbzh_found = True

                            # Verify DBZH structure
                            dbzh_config = value["variables"]["DBZH"]
                            assert (
                                "units" in dbzh_config["attributes"]
                            ), f"{vcp_name} DBZH missing units"
                            assert (
                                dbzh_config["attributes"]["units"] == "dBZ"
                            ), f"{vcp_name} DBZH wrong units"

        assert dbzh_found, "DBZH variable not found in any IDEAM VCP"


class TestEcccConfigValidation:
    """Test ECCC config file validation."""

    @pytest.fixture
    def eccc_config(self):
        """Load ECCC config for testing."""
        try:
            manager = VcpTemplateManager("eccc.json")
            return manager.config
        except FileNotFoundError:
            pytest.skip("ECCC config file not available")

    def test_eccc_default_vcp(self, eccc_config):
        """Test that ECCC config has DEFAULT VCP."""
        assert "DEFAULT" in eccc_config, "ECCC config missing DEFAULT VCP"

    def test_eccc_default_structure(self, eccc_config):
        """Test that ECCC DEFAULT VCP has correct structure."""
        default_config = eccc_config["DEFAULT"]

        # Required fields
        assert "elevations" in default_config, "DEFAULT missing elevations"
        assert "dims" in default_config, "DEFAULT missing dims"

        # Should have 17 sweeps for ECCC
        elevations = default_config["elevations"]
        assert (
            len(elevations) == 17
        ), f"ECCC DEFAULT should have 17 sweeps, got {len(elevations)}"

        # Check all sweep configurations exist
        for i in range(17):
            sweep_key = f"sweep_{i}"
            assert sweep_key in default_config, f"DEFAULT missing {sweep_key}"

    def test_eccc_dual_pol_variables(self, eccc_config):
        """Test that ECCC has dual-polarization variables."""
        default_config = eccc_config["DEFAULT"]

        # Expected ECCC variables
        expected_vars = [
            "DBZH",
            "TH",
            "RHOHV",
            "UPHIDP",
            "WRADH",
            "PHIDP",
            "ZDR",
            "KDP",
            "SQIH",
            "VRADH",
        ]

        # Check first sweep has all expected variables
        sweep_0 = default_config["sweep_0"]
        assert "variables" in sweep_0, "sweep_0 missing variables"

        variables = sweep_0["variables"]
        for var in expected_vars:
            assert var in variables, f"sweep_0 missing variable: {var}"

            # Check variable structure
            var_config = variables[var]
            assert "dtype" in var_config, f"{var} missing dtype"
            assert "fill_value" in var_config, f"{var} missing fill_value"
            assert "attributes" in var_config, f"{var} missing attributes"

            # Check units exist
            attrs = var_config["attributes"]
            assert "units" in attrs, f"{var} missing units in attributes"

    def test_eccc_dimension_consistency(self, eccc_config):
        """Test that ECCC dimensions are consistent."""
        default_config = eccc_config["DEFAULT"]
        dims = default_config["dims"]

        assert "azimuth" in dims, "DEFAULT missing azimuth dims"
        assert "range" in dims, "DEFAULT missing range dims"

        # Should have dimensions for all 17 sweeps
        assert len(dims["azimuth"]) == 17, "azimuth dims count mismatch"
        assert len(dims["range"]) == 17, "range dims count mismatch"

        # All dimensions should be positive integers
        for i, (az, rng) in enumerate(zip(dims["azimuth"], dims["range"])):
            assert (
                isinstance(az, int) and az > 0
            ), f"Invalid azimuth dim at sweep {i}: {az}"
            assert (
                isinstance(rng, int) and rng > 0
            ), f"Invalid range dim at sweep {i}: {rng}"


class TestConfigConsistency:
    """Test consistency across different config files."""

    def test_variable_structure_consistency(self):
        """Test that all configs use consistent variable structure."""
        configs = {}
        config_files = ["vcp_nexrad.json", "ideam.json", "eccc.json"]

        for config_file in config_files:
            try:
                manager = VcpTemplateManager(config_file)
                configs[config_file] = manager.config
            except FileNotFoundError:
                continue  # Skip missing configs

        # If we have multiple configs, check structure consistency
        if len(configs) > 1:
            for config_name, config in configs.items():
                for vcp_name, vcp_config in config.items():
                    # Check top-level structure
                    assert (
                        "elevations" in vcp_config
                    ), f"{config_name} {vcp_name} missing elevations"
                    assert (
                        "dims" in vcp_config
                    ), f"{config_name} {vcp_name} missing dims"

                    # Check sweep structure
                    for key, value in vcp_config.items():
                        if key.startswith("sweep_") and isinstance(value, dict):
                            if "variables" in value:
                                for var_name, var_config in value["variables"].items():
                                    # All variables should have dtype at minimum
                                    assert (
                                        "dtype" in var_config
                                    ), f"{config_name} {vcp_name} {key} {var_name} missing dtype"

                                    # Check that dtype is valid
                                    valid_dtypes = [
                                        "float32",
                                        "float64",
                                        "int16",
                                        "int32",
                                        "int64",
                                        "str",
                                    ]
                                    assert (
                                        var_config["dtype"] in valid_dtypes
                                    ), f"{config_name} {vcp_name} {key} {var_name} invalid dtype: {var_config['dtype']}"

    def test_elevation_angle_ranges(self):
        """Test that elevation angles are within reasonable ranges."""
        config_files = ["vcp_nexrad.json", "ideam.json", "eccc.json"]

        for config_file in config_files:
            try:
                manager = VcpTemplateManager(config_file)
                config = manager.config

                for vcp_name, vcp_config in config.items():
                    elevations = vcp_config["elevations"]

                    for i, elev in enumerate(elevations):
                        assert (
                            0 <= elev <= 90
                        ), f"{config_file} {vcp_name} elevation {i} out of range: {elev}"

                        # Check elevation progression (can vary for different radar modes)
                        if i > 0:
                            prev_elev = elevations[i - 1]
                            # Allow flexibility as different radar modes may have non-monotonic sequences
                            # Only check for reasonable jumps (no more than 30 degrees)
                            assert (
                                abs(elev - prev_elev) <= 30.0
                            ), f"{config_file} {vcp_name} elevation jump too large at {i}: {prev_elev} -> {elev}"

            except FileNotFoundError:
                continue  # Skip missing configs

    def test_dimension_reasonableness(self):
        """Test that azimuth and range dimensions are reasonable."""
        config_files = ["vcp_nexrad.json", "ideam.json", "eccc.json"]

        for config_file in config_files:
            try:
                manager = VcpTemplateManager(config_file)
                config = manager.config

                for vcp_name, vcp_config in config.items():
                    if "dims" in vcp_config:
                        dims = vcp_config["dims"]

                        if "azimuth" in dims:
                            for i, az in enumerate(dims["azimuth"]):
                                # Azimuth should be reasonable radar values
                                assert (
                                    180 <= az <= 1440
                                ), f"{config_file} {vcp_name} unreasonable azimuth at {i}: {az}"

                        if "range" in dims:
                            for i, rng in enumerate(dims["range"]):
                                # Range should be reasonable radar values
                                assert (
                                    100 <= rng <= 5000
                                ), f"{config_file} {vcp_name} unreasonable range at {i}: {rng}"

            except FileNotFoundError:
                continue  # Skip missing configs


class TestTemplateManagerIntegration:
    """Test VcpTemplateManager integration with different configs."""

    def test_config_switching(self):
        """Test that VcpTemplateManager can switch between configs."""
        config_files = ["vcp_nexrad.json", "ideam.json", "eccc.json"]

        for config_file in config_files:
            try:
                manager = VcpTemplateManager(config_file)
                assert manager.config is not None, f"Failed to load {config_file}"

                # Test that config path is correct
                assert config_file in str(
                    manager.vcp_nexrad_path
                ), f"Config path incorrect for {config_file}"

            except FileNotFoundError:
                # Config file not available, skip
                continue

    def test_vcp_info_retrieval(self):
        """Test VCP info retrieval for different configs."""
        # Test IDEAM configs
        try:
            ideam_manager = VcpTemplateManager("ideam.json")
            vcp_names = ["PRECC", "SURVP", "PRECA", "PRECB"]

            for vcp_name in vcp_names:
                if vcp_name in ideam_manager.config:
                    vcp_info = ideam_manager.get_vcp_info(vcp_name)
                    assert vcp_info is not None
                    assert len(vcp_info.elevations) > 0
                    assert len(vcp_info.dims) > 0

        except FileNotFoundError:
            pass  # Skip if IDEAM config not available

        # Test ECCC config
        try:
            eccc_manager = VcpTemplateManager("eccc.json")
            if "DEFAULT" in eccc_manager.config:
                vcp_info = eccc_manager.get_vcp_info("DEFAULT")
                assert vcp_info is not None
                assert len(vcp_info.elevations) == 17  # ECCC has 17 sweeps

        except FileNotFoundError:
            pass  # Skip if ECCC config not available

    def test_sweep_config_retrieval(self):
        """Test sweep configuration retrieval."""
        try:
            ideam_manager = VcpTemplateManager("ideam.json")
            if "PRECC" in ideam_manager.config:
                sweep_config = ideam_manager.get_sweep_config("PRECC", 0)
                assert "variables" in sweep_config
                assert isinstance(sweep_config["variables"], dict)

        except (FileNotFoundError, ValueError):
            pass  # Skip if not available

        try:
            eccc_manager = VcpTemplateManager("eccc.json")
            if "DEFAULT" in eccc_manager.config:
                sweep_config = eccc_manager.get_sweep_config("DEFAULT", 0)
                assert "variables" in sweep_config
                assert isinstance(sweep_config["variables"], dict)

        except (FileNotFoundError, ValueError):
            pass  # Skip if not available
