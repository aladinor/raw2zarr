import json
from pathlib import Path

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
        """Test that VCP-12 template matches VCP configuration from config/vcp.json."""
        append_dim_time = [pd.Timestamp("2023-01-01 12:00:00")]

        tree = template_manager.create_empty_vcp_tree(
            radar_info=radar_info,
            append_dim="vcp_time",
            append_dim_time=append_dim_time,
        )

        # Load VCP-12 configuration
        config_path = (
            Path(__file__).parent.parent.parent / "raw2zarr" / "config" / "vcp.json"
        )
        with open(config_path) as f:
            vcp_config = json.load(f)

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
