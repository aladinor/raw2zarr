"""Unit tests for template operations."""

import pandas as pd
import pytest
import xarray as xr
from xarray import DataTree

from raw2zarr.templates.template_ops import (
    create_vcp_template_in_memory,
    merge_data_into_template,
)


class TestCreateVcpTemplateInMemory:
    """Test suite for create_vcp_template_in_memory function."""

    def test_creates_valid_datatree(self):
        """Test that function creates a valid DataTree structure."""
        template = create_vcp_template_in_memory(
            vcp="VCP-212", append_dim="vcp_time", vcp_config_file="vcp_nexrad.json"
        )

        assert isinstance(template, DataTree)
        assert len(template.groups) > 0

    def test_has_vcp_root_group(self):
        """Test that template has VCP root group."""
        template = create_vcp_template_in_memory(vcp="VCP-212", append_dim="vcp_time")

        # Should have VCP-212 as a group
        assert "/VCP-212" in template.groups

    def test_has_sweep_groups(self):
        """Test that template has sweep groups."""
        template = create_vcp_template_in_memory(vcp="VCP-212", append_dim="vcp_time")

        # VCP-212 should have sweeps
        sweep_groups = [g for g in template.groups if "sweep" in g.lower()]
        assert len(sweep_groups) > 0

    def test_append_dim_exists(self):
        """Test that append dimension exists in datasets."""
        append_dim = "vcp_time"
        template = create_vcp_template_in_memory(vcp="VCP-212", append_dim=append_dim)

        # Check VCP root has append_dim
        vcp_ds = template["/VCP-212"].ds
        assert append_dim in vcp_ds.dims or append_dim in vcp_ds.coords

    def test_uses_custom_config_file(self):
        """Test that custom VCP config file can be specified."""
        # This should not raise an error
        template = create_vcp_template_in_memory(
            vcp="VCP-212",
            append_dim="vcp_time",
            vcp_config_file="vcp_nexrad.json",
        )
        assert isinstance(template, DataTree)

    def test_different_vcp_patterns(self):
        """Test creating templates for different VCP patterns."""
        vcps = ["VCP-212", "VCP-32", "VCP-215"]

        for vcp in vcps:
            template = create_vcp_template_in_memory(vcp=vcp, append_dim="vcp_time")
            assert isinstance(template, DataTree)
            assert (
                f"/VCP-{vcp.split('-')[1]}" in template.groups
                or f"/{vcp}" in template.groups
            )


class TestMergeDataIntoTemplate:
    """Test suite for merge_data_into_template function."""

    @pytest.fixture
    def simple_template(self):
        """Create a simple template DataTree for testing."""
        # Create a simple template with VCP root and one sweep
        vcp_ds = xr.Dataset(
            {
                "volume_number": (["vcp_time"], [0]),
                "latitude": (["vcp_time"], [35.0]),
                "longitude": (["vcp_time"], [-97.0]),
            },
            coords={"vcp_time": [pd.Timestamp("2020-01-01")]},
        )

        sweep_ds = xr.Dataset(
            {
                "DBZH": (
                    ["vcp_time", "azimuth", "range"],
                    [[[float("nan")] * 10] * 5],
                ),
                "VRADH": (
                    ["vcp_time", "azimuth", "range"],
                    [[[float("nan")] * 10] * 5],
                ),
            },
            coords={
                "vcp_time": [pd.Timestamp("2020-01-01")],
                "azimuth": range(5),
                "range": range(10),
            },
        )

        return DataTree.from_dict(
            {
                "/VCP-212": vcp_ds,
                "/VCP-212/sweep_0": sweep_ds,
            }
        )

    @pytest.fixture
    def simple_actual_data(self):
        """Create simple actual data with missing variable."""
        # Actual data only has DBZH, missing VRADH
        vcp_ds = xr.Dataset(
            {
                "volume_number": (["vcp_time"], [1]),
                "latitude": (["vcp_time"], [35.5]),
                "longitude": (["vcp_time"], [-97.5]),
            },
            coords={"vcp_time": [pd.Timestamp("2020-01-01")]},
        )

        sweep_ds = xr.Dataset(
            {
                "DBZH": (
                    ["vcp_time", "azimuth", "range"],
                    [[[1.0] * 10] * 5],
                ),  # Has data
                # VRADH is missing
            },
            coords={
                "vcp_time": [pd.Timestamp("2020-01-01")],
                "azimuth": range(5),
                "range": range(10),
            },
        )

        return DataTree.from_dict(
            {
                "/VCP-212": vcp_ds,
                "/VCP-212/sweep_0": sweep_ds,
            }
        )

    def test_returns_datatree(self, simple_template, simple_actual_data):
        """Test that merge returns a DataTree."""
        result = merge_data_into_template(simple_template, simple_actual_data)
        assert isinstance(result, DataTree)

    def test_preserves_actual_data_values(self, simple_template, simple_actual_data):
        """Test that actual data values are preserved."""
        result = merge_data_into_template(simple_template, simple_actual_data)

        # DBZH should have actual values (1.0), not template values (NaN)
        dbzh_values = result["/VCP-212/sweep_0"].ds["DBZH"].values
        assert dbzh_values[0, 0, 0] == 1.0  # Actual data value

    def test_adds_missing_variables_from_template(
        self, simple_template, simple_actual_data
    ):
        """Test that missing variables are added from template."""
        result = merge_data_into_template(simple_template, simple_actual_data)

        # VRADH should be present (from template) even though missing in actual
        assert "VRADH" in result["/VCP-212/sweep_0"].ds.data_vars

    def test_missing_variables_have_nan_values(
        self, simple_template, simple_actual_data
    ):
        """Test that missing variables have NaN values from template."""
        result = merge_data_into_template(simple_template, simple_actual_data)

        # VRADH should have NaN values (from template)
        import numpy as np

        vradh_values = result["/VCP-212/sweep_0"].ds["VRADH"].values
        assert np.all(np.isnan(vradh_values))

    def test_preserves_group_structure(self, simple_template, simple_actual_data):
        """Test that group structure is preserved."""
        result = merge_data_into_template(simple_template, simple_actual_data)

        # Should have same groups
        assert "/VCP-212" in result.groups
        assert "/VCP-212/sweep_0" in result.groups

    def test_handles_missing_sweep_in_actual(self, simple_template):
        """Test handling when actual data is missing entire sweep."""
        # Actual data with no sweep_0
        actual_data = DataTree.from_dict(
            {
                "/VCP-212": xr.Dataset(
                    {"volume_number": (["vcp_time"], [1])},
                    coords={"vcp_time": [pd.Timestamp("2020-01-01")]},
                ),
            }
        )

        result = merge_data_into_template(simple_template, actual_data)

        # sweep_0 should be present from template
        assert "/VCP-212/sweep_0" in result.groups

    def test_handles_paths_with_and_without_leading_slash(self):
        """Test that path normalization works correctly."""
        template_dict = {
            "VCP-212": xr.Dataset({"var1": ([], 1.0)}),
        }
        actual_dict = {
            "/VCP-212": xr.Dataset({"var1": ([], 2.0)}),
        }

        template = DataTree.from_dict(template_dict)
        actual = DataTree.from_dict(actual_dict)

        # Should not raise error due to path mismatch
        result = merge_data_into_template(template, actual)
        assert isinstance(result, DataTree)

    def test_empty_actual_data_returns_template(self, simple_template):
        """Test that empty actual data returns full template."""
        # Empty actual data
        actual_data = DataTree.from_dict({"/VCP-212": xr.Dataset()})

        result = merge_data_into_template(simple_template, actual_data)

        # Should have all template groups and variables
        assert "/VCP-212/sweep_0" in result.groups
        assert "DBZH" in result["/VCP-212/sweep_0"].ds.data_vars
        assert "VRADH" in result["/VCP-212/sweep_0"].ds.data_vars
