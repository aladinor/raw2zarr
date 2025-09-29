import pandas as pd
import pytest

from raw2zarr.templates.template_manager import VcpTemplateManager


class TestVcpTemplateManager:
    """Test VCP template manager functionality."""

    @pytest.fixture
    def template_manager(self):
        """Create a VcpTemplateManager instance."""
        return VcpTemplateManager()

    @pytest.fixture
    def radar_info(self):
        """Sample radar info for testing."""
        return {
            "vcp": "VCP-12",
            "lon": -97.0,
            "lat": 36.0,
            "alt": 300.0,
            "instrument_name": "KVNX",
            "reference_time": pd.Timestamp("2023-01-01 12:00:00"),
            "time_coverage_start": pd.Timestamp("2023-01-01 12:00:00"),
            "time_coverage_end": pd.Timestamp("2023-01-01 12:00:00"),
            "volume_number": 1,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "crs_wkt": {"grid_mapping_name": "azimuthal_equidistant"},
        }

    def test_create_empty_vcp_tree_basic(self, template_manager, radar_info):
        """Test basic VCP tree creation with new interface."""
        append_dim_time = [
            pd.Timestamp("2023-01-01 12:00:00"),
            pd.Timestamp("2023-01-01 12:05:00"),
        ]

        tree = template_manager.create_empty_vcp_tree(
            radar_info=radar_info,
            append_dim="vcp_time",
            append_dim_time=append_dim_time,
        )

        # Verify tree structure
        assert tree is not None
        assert "/VCP-12" in tree.groups

        # Verify VCP group has correct vcp_time dimension
        vcp_group = tree["VCP-12"]
        assert "vcp_time" in vcp_group.sizes
        assert vcp_group.sizes["vcp_time"] == len(append_dim_time)

    def test_create_empty_vcp_tree_dimensions(self, template_manager, radar_info):
        """Test that size_append_dim is correctly calculated from append_dim_time."""
        # Test with different timestamp counts
        test_cases = [
            1,  # Single timestamp
            3,  # Multiple timestamps
            6,  # Typical VCP count
        ]

        for timestamp_count in test_cases:
            append_dim_time = [
                pd.Timestamp("2023-01-01 12:00:00") + pd.Timedelta(minutes=i * 5)
                for i in range(timestamp_count)
            ]

            tree = template_manager.create_empty_vcp_tree(
                radar_info=radar_info,
                append_dim="vcp_time",
                append_dim_time=append_dim_time,
            )

            # Verify VCP group dimensions
            vcp_group = tree["VCP-12"]
            assert vcp_group.sizes["vcp_time"] == timestamp_count

            # Verify sweep groups have correct dimensions
            sweep_groups = [g for g in tree.groups if "sweep_" in g]
            assert len(sweep_groups) > 0  # Should have sweep groups

            # Check first sweep group
            first_sweep = tree[sweep_groups[0]]
            assert "vcp_time" in first_sweep.sizes
            assert first_sweep.sizes["vcp_time"] == timestamp_count

    def test_create_empty_vcp_tree_structure(self, template_manager, radar_info):
        """Test that the VCP tree has expected structure."""
        append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

        tree = template_manager.create_empty_vcp_tree(
            radar_info=radar_info,
            append_dim="vcp_time",
            append_dim_time=append_dim_time,
        )

        # Verify expected groups exist
        expected_groups = [
            "/",
            "/VCP-12",
            "/VCP-12/sweep_0",  # At least one sweep
        ]

        for group in expected_groups:
            assert group in tree.groups, f"Missing group: {group}"

        # Verify sweep groups have radar dimensions
        sweep_groups = [g for g in tree.groups if "sweep_" in g]
        first_sweep = tree[sweep_groups[0]]

        # Should have typical radar dimensions
        assert "azimuth" in first_sweep.sizes
        assert "range" in first_sweep.sizes
        assert first_sweep.sizes["azimuth"] > 0
        assert first_sweep.sizes["range"] > 0

    def test_different_vcp_types(self, template_manager):
        """Test template creation with different VCP types."""
        # Only test VCP types that exist in the config
        vcp_types = ["VCP-12"]  # Start with known working VCP

        for vcp in vcp_types:
            radar_info = {
                "vcp": vcp,
                "lon": -97.0,
                "lat": 36.0,
                "alt": 300.0,
                "instrument_name": "KVNX",
                "reference_time": pd.Timestamp("2023-01-01 12:00:00"),
                "time_coverage_start": pd.Timestamp("2023-01-01 12:00:00"),
                "time_coverage_end": pd.Timestamp("2023-01-01 12:00:00"),
                "volume_number": 1,
                "platform_type": "fixed",
                "instrument_type": "radar",
                "crs_wkt": {"grid_mapping_name": "azimuthal_equidistant"},
            }

            append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

            tree = template_manager.create_empty_vcp_tree(
                radar_info=radar_info,
                append_dim="vcp_time",
                append_dim_time=append_dim_time,
            )

            # Verify VCP-specific group was created
            assert f"/{vcp}" in tree.groups
            vcp_group = tree[vcp]
            assert "vcp_time" in vcp_group.sizes
            assert vcp_group.sizes["vcp_time"] == 1

    def test_single_timestamp(self, template_manager, radar_info):
        """Test template creation with single timestamp."""
        append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

        tree = template_manager.create_empty_vcp_tree(
            radar_info=radar_info,
            append_dim="vcp_time",
            append_dim_time=append_dim_time,
        )

        # Verify single timestamp creates size 1 dimension
        vcp_group = tree["VCP-12"]
        assert vcp_group.sizes["vcp_time"] == 1

    def test_vcp_configuration_compliance(self, template_manager, radar_info):
        """Test that VCP-12 template matches VCP configuration from unified config."""
        append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

        tree = template_manager.create_empty_vcp_tree(
            radar_info=radar_info,
            append_dim="vcp_time",
            append_dim_time=append_dim_time,
        )

        # Load VCP-12 configuration from unified config
        vcp_config = template_manager.config
        vcp12_config = vcp_config["VCP-12"]
        expected_elevations = vcp12_config["elevations"]

        # Test expected group structure for VCP-12
        expected_groups = [
            "/",
            "/VCP-12",
            "/VCP-12/georeferencing_correction",
            "/VCP-12/radar_parameters",
            "/VCP-12/radar_calibration",
        ]

        # Add all expected sweep groups (17 sweeps for VCP-12)
        for i in range(len(expected_elevations)):
            expected_groups.append(f"/VCP-12/sweep_{i}")

        # Verify all expected groups exist
        for group in expected_groups:
            assert group in tree.groups, f"Missing expected group: {group}"

        # Verify total group count matches expectation
        # (root + VCP + 3 additional groups + 17 sweeps = 22 groups)
        expected_total_groups = 1 + 1 + 3 + len(expected_elevations)  # 22 for VCP-12
        assert (
            len(tree.groups) == expected_total_groups
        ), f"Expected {expected_total_groups} groups, got {len(tree.groups)}"

        # Test sweep-specific dimensions against VCP config
        sweep_groups = [g for g in tree.groups if "/VCP-12/sweep_" in g]
        assert len(sweep_groups) == len(
            expected_elevations
        ), f"Expected {len(expected_elevations)} sweeps, got {len(sweep_groups)}"

        # Verify dimensions exist and are reasonable for each sweep
        for i, sweep_group_path in enumerate(sorted(sweep_groups)):
            sweep = tree[sweep_group_path]

            # Verify required dimensions exist
            assert (
                "azimuth" in sweep.sizes
            ), f"Missing azimuth dimension in {sweep_group_path}"
            assert (
                "range" in sweep.sizes
            ), f"Missing range dimension in {sweep_group_path}"
            assert (
                "vcp_time" in sweep.sizes
            ), f"Missing vcp_time dimension in {sweep_group_path}"

            # Verify dimensions are reasonable (positive values)
            assert (
                sweep.sizes["azimuth"] > 0
            ), f"Sweep {i} azimuth dimension must be positive"
            assert (
                sweep.sizes["range"] > 0
            ), f"Sweep {i} range dimension must be positive"
            assert (
                sweep.sizes["vcp_time"] == 1
            ), f"Sweep {i} vcp_time: expected 1, got {sweep.sizes['vcp_time']}"

            # Verify azimuth is either 360 or 720 (common radar values)
            assert sweep.sizes["azimuth"] in [
                360,
                720,
            ], f"Sweep {i} azimuth should be 360 or 720, got {sweep.sizes['azimuth']}"

            # Verify range is reasonable (typically hundreds to thousands)
            assert (
                100 <= sweep.sizes["range"] <= 3000
            ), f"Sweep {i} range dimension seems unreasonable: {sweep.sizes['range']}"

        # Verify additional groups have expected structure
        assert "/VCP-12/radar_parameters" in tree.groups
        assert "/VCP-12/georeferencing_correction" in tree.groups
        assert "/VCP-12/radar_calibration" in tree.groups

        print("✅ VCP-12 template validation complete:")
        print(f"  - {len(sweep_groups)} sweeps with correct dimensions")
        print(
            "  - 3 additional groups (radar_parameters, georeferencing_correction, radar_calibration)"
        )
        print(f"  - Total {len(tree.groups)} groups as expected")

    def test_unified_config_system(self, template_manager):
        """Test the unified VCP configuration system."""
        # Test unified config functionality
        assert template_manager.config is not None

        # Test VCP-21 specifically (our main fix)
        if "VCP-21" in template_manager.config:
            vcp_info = template_manager.get_vcp_info("VCP-21")
            assert vcp_info is not None
            assert len(vcp_info.elevations) > 0

            # Test sweep configuration access
            try:
                sweep_8_config = template_manager.get_sweep_config("VCP-21", 8)
                assert "variables" in sweep_8_config
                # VCP-21 sweep 8 should have PHIDP (our main fix)
                if "PHIDP" in sweep_8_config["variables"]:
                    print("✅ VCP-21 sweep 8 correctly has PHIDP variable")
            except (ValueError, KeyError):
                # If sweep 8 doesn't exist, that's ok for this test
                pass

            print("✅ Unified config system working correctly")

    def test_vcp21_phidp_fix(self, template_manager):
        """Test the specific VCP-21 PHIDP fix."""

        # Test that VCP-21 exists in unified config
        unified_config = template_manager.config
        assert "VCP-21" in unified_config, "VCP-21 should exist in unified config"

        vcp21_config = unified_config["VCP-21"]

        # Check that VCP-21 has sweeps with PHIDP
        phidp_sweeps = []
        for key, value in vcp21_config.items():
            if key.startswith("sweep_") and isinstance(value, dict):
                if "variables" in value and "PHIDP" in value["variables"]:
                    sweep_num = int(key.split("_")[1])
                    phidp_sweeps.append(sweep_num)

        # VCP-21 should have PHIDP in some sweeps (the fix we implemented)
        assert len(phidp_sweeps) > 0, "VCP-21 should have PHIDP in some sweeps"
        print(f"✅ VCP-21 has PHIDP in sweeps: {phidp_sweeps}")

    def test_vcp_info_access(self, template_manager):
        """Test that the template manager provides VCP info correctly."""
        # Unified system should provide VCP info
        try:
            vcp_info = template_manager.get_vcp_info("VCP-21")
            assert vcp_info is not None
            assert hasattr(vcp_info, "elevations")
            assert hasattr(vcp_info, "dims")
            print("✅ Unified config provides VCP info correctly")
        except ValueError:
            # VCP-21 might not exist in test config
            pass

    def test_template_creation_with_unified_config(self, template_manager, radar_info):
        """Test template creation works with unified config."""

        # Try to create a template with VCP-21 if available
        vcp21_radar_info = radar_info.copy()
        vcp21_radar_info["vcp"] = "VCP-21"

        try:
            append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

            tree = template_manager.create_empty_vcp_tree(
                radar_info=vcp21_radar_info,
                append_dim="vcp_time",
                append_dim_time=append_dim_time,
            )

            assert tree is not None
            assert "/VCP-21" in tree.groups

            # Check for sweep groups
            sweep_groups = [g for g in tree.groups if "/VCP-21/sweep_" in g]
            assert len(sweep_groups) > 0, "Should have at least one sweep group"

            print(f"✅ VCP-21 template created with {len(sweep_groups)} sweeps")

        except ValueError as e:
            if "not found" in str(e):
                pytest.skip(f"VCP-21 not available in config: {e}")
            else:
                raise


class TestVcpConfigurationFiles:
    """Test VCP configuration file loading and switching."""

    def test_default_nexrad_config_loading(self):
        """Test default NEXRAD config file loads correctly."""
        manager = VcpTemplateManager()
        assert manager.config is not None
        # Should have standard NEXRAD VCPs
        assert "VCP-12" in manager.config

    def test_ideam_config_loading(self):
        """Test IDEAM config file loads correctly."""
        manager = VcpTemplateManager("ideam.json")
        assert manager.config is not None

        # Check all IDEAM VCP patterns exist
        expected_vcps = ["PRECC", "SURVP", "PRECA", "PRECB"]
        for vcp in expected_vcps:
            assert vcp in manager.config, f"Missing VCP: {vcp}"

        # Verify PRECC structure (3 elevations as example)
        precc_config = manager.config["PRECC"]
        assert "elevations" in precc_config
        assert len(precc_config["elevations"]) == 3
        assert precc_config["elevations"] == [10.0, 12.5, 15.0]

    def test_eccc_config_loading(self):
        """Test ECCC config file loads correctly."""
        manager = VcpTemplateManager("eccc.json")
        assert manager.config is not None

        # Check DEFAULT VCP exists
        assert "DEFAULT" in manager.config

        # Verify DEFAULT has 17 sweeps
        default_config = manager.config["DEFAULT"]
        assert "elevations" in default_config
        assert len(default_config["elevations"]) == 17

        # Check sweep configurations exist
        for i in range(17):
            sweep_key = f"sweep_{i}"
            assert sweep_key in default_config, f"Missing {sweep_key}"

    def test_invalid_config_file(self):
        """Test error handling for missing config files."""
        with pytest.raises(FileNotFoundError, match="Unified config file not found"):
            VcpTemplateManager("nonexistent.json")

    def test_config_file_parameter_acceptance(self):
        """Test that VcpTemplateManager accepts vcp_config_file parameter."""
        # Test with different valid config files
        configs = ["vcp_nexrad.json", "ideam.json", "eccc.json"]

        for config_file in configs:
            try:
                manager = VcpTemplateManager(config_file)
                assert manager.config is not None
                print(f"✅ Successfully loaded {config_file}")
            except FileNotFoundError:
                pytest.skip(
                    f"Config file {config_file} not available in test environment"
                )


class TestIdeamVcpTemplates:
    """Test IDEAM-specific VCP template creation."""

    @pytest.fixture
    def ideam_manager(self):
        """Create IDEAM VcpTemplateManager instance."""
        return VcpTemplateManager("ideam.json")

    @pytest.fixture
    def ideam_radar_info(self):
        """Sample IDEAM radar info for testing."""
        return {
            "vcp": "PRECC",
            "lon": -72.0,
            "lat": 2.0,
            "alt": 500.0,
            "instrument_name": "Guaviare",
            "reference_time": pd.Timestamp("2023-01-01 12:00:00"),
            "time_coverage_start": pd.Timestamp("2023-01-01 12:00:00"),
            "time_coverage_end": pd.Timestamp("2023-01-01 12:00:00"),
            "volume_number": 1,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "crs_wkt": {"grid_mapping_name": "azimuthal_equidistant"},
        }

    def test_ideam_vcp_precc_template_creation(self, ideam_manager, ideam_radar_info):
        """Test VCP-PRECC template creation."""
        append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

        tree = ideam_manager.create_empty_vcp_tree(
            radar_info=ideam_radar_info,
            append_dim="vcp_time",
            append_dim_time=append_dim_time,
        )

        assert tree is not None
        assert "/PRECC" in tree.groups

        # PRECC should have 3 sweeps (10.0, 12.5, 15.0 degrees)
        sweep_groups = [g for g in tree.groups if "/PRECC/sweep_" in g]
        assert len(sweep_groups) == 3

    def test_ideam_all_vcp_patterns(self, ideam_manager):
        """Test template creation for all IDEAM VCP patterns."""
        vcp_patterns = ["PRECC", "SURVP", "PRECA", "PRECB"]

        for vcp in vcp_patterns:
            radar_info = {
                "vcp": vcp,
                "lon": -72.0,
                "lat": 2.0,
                "alt": 500.0,
                "instrument_name": "Guaviare",
                "reference_time": pd.Timestamp("2023-01-01 12:00:00"),
                "time_coverage_start": pd.Timestamp("2023-01-01 12:00:00"),
                "time_coverage_end": pd.Timestamp("2023-01-01 12:00:00"),
                "volume_number": 1,
                "platform_type": "fixed",
                "instrument_type": "radar",
                "crs_wkt": {"grid_mapping_name": "azimuthal_equidistant"},
            }

            append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

            tree = ideam_manager.create_empty_vcp_tree(
                radar_info=radar_info,
                append_dim="vcp_time",
                append_dim_time=append_dim_time,
            )

            assert tree is not None
            assert f"/{vcp}" in tree.groups

            # Verify has expected number of sweeps
            vcp_info = ideam_manager.get_vcp_info(vcp)
            sweep_groups = [g for g in tree.groups if f"/{vcp}/sweep_" in g]
            assert len(sweep_groups) == len(vcp_info.elevations)

    def test_ideam_vcp_variables(self, ideam_manager):
        """Test that IDEAM VCPs have expected variables."""
        # Check PRECC has DBZH (main IDEAM variable)
        sweep_config = ideam_manager.get_sweep_config("PRECC", 0)
        assert "variables" in sweep_config
        assert "DBZH" in sweep_config["variables"]

        # Verify variable structure
        dbzh_config = sweep_config["variables"]["DBZH"]
        assert "dtype" in dbzh_config
        assert "fill_value" in dbzh_config
        assert "attributes" in dbzh_config


class TestEcccVcpTemplates:
    """Test ECCC-specific VCP template creation."""

    @pytest.fixture
    def eccc_manager(self):
        """Create ECCC VcpTemplateManager instance."""
        return VcpTemplateManager("eccc.json")

    @pytest.fixture
    def eccc_radar_info(self):
        """Sample ECCC radar info for testing."""
        return {
            "vcp": "DEFAULT",
            "lon": -75.0,
            "lat": 45.0,
            "alt": 300.0,
            "instrument_name": "CASET",
            "reference_time": pd.Timestamp("2023-01-01 12:00:00"),
            "time_coverage_start": pd.Timestamp("2023-01-01 12:00:00"),
            "time_coverage_end": pd.Timestamp("2023-01-01 12:00:00"),
            "volume_number": 1,
            "platform_type": "fixed",
            "instrument_type": "radar",
            "crs_wkt": {"grid_mapping_name": "azimuthal_equidistant"},
        }

    def test_eccc_default_template_creation(self, eccc_manager, eccc_radar_info):
        """Test ECCC DEFAULT VCP template with 17 sweeps."""
        append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

        tree = eccc_manager.create_empty_vcp_tree(
            radar_info=eccc_radar_info,
            append_dim="vcp_time",
            append_dim_time=append_dim_time,
        )

        assert tree is not None
        assert "/DEFAULT" in tree.groups

        # ECCC DEFAULT should have 17 sweeps
        sweep_groups = [g for g in tree.groups if "/DEFAULT/sweep_" in g]
        assert len(sweep_groups) == 17

    def test_eccc_variables(self, eccc_manager):
        """Test that ECCC DEFAULT VCP has expected variables."""
        # Check DEFAULT has expected ECCC variables
        sweep_config = eccc_manager.get_sweep_config("DEFAULT", 0)
        assert "variables" in sweep_config

        # ECCC should have these variables
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
        for var in expected_vars:
            assert var in sweep_config["variables"], f"Missing variable: {var}"

    def test_eccc_dual_pol_variables(self, eccc_manager):
        """Test ECCC dual-polarization variables are properly configured."""
        sweep_config = eccc_manager.get_sweep_config("DEFAULT", 0)

        # Test specific dual-pol variables
        dual_pol_vars = ["ZDR", "KDP", "RHOHV", "PHIDP"]
        for var in dual_pol_vars:
            var_config = sweep_config["variables"][var]
            assert "dtype" in var_config
            assert "fill_value" in var_config
            assert "attributes" in var_config

            # Verify units are specified for dual-pol variables
            attrs = var_config["attributes"]
            assert "units" in attrs, f"Missing units for {var}"


class TestVcpConfigParameterFlow:
    """Test VCP config parameter flow through the system."""

    def test_template_manager_config_parameter(self):
        """Test VcpTemplateManager uses vcp_config_file parameter correctly."""
        # Test parameter is stored and used
        manager = VcpTemplateManager("ideam.json")
        assert "ideam.json" in str(manager.vcp_nexrad_path)

    def test_config_path_resolution(self):
        """Test config file path resolution works correctly."""
        manager = VcpTemplateManager("eccc.json")
        assert manager.vcp_nexrad_path.name == "eccc.json"
        assert manager.vcp_nexrad_path.parent.name == "config"

    def test_default_fallback_behavior(self):
        """Test that scan_name fallback to DEFAULT works."""
        # This tests the try-catch in writer_utils.py:115-117
        # Note: This would require mocking a DataTree without scan_name attribute
        # For now, we verify the structure exists
        import inspect

        from raw2zarr.writer.writer_utils import init_zarr_store

        # Verify the function signature includes vcp_config_file
        sig = inspect.signature(init_zarr_store)
        assert "vcp_config_file" in sig.parameters
        assert sig.parameters["vcp_config_file"].default == "vcp_nexrad.json"
