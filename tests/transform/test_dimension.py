import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataTree

from raw2zarr.transform.dimension import (
    ensure_dimension,
    exp_dim,
    slice_to_vcp_dimensions,
)


class TestDimensionTransforms:
    def create_dtree_without_dim(self) -> DataTree:
        root_data = xr.Dataset(
            data_vars={"time_coverage_start": pd.Timestamp("2023-01-01T00:00:00")}
        )

        ds = xr.Dataset(
            data_vars={"reflectivity": (["azimuth", "range"], np.random.rand(5, 10))},
            coords={
                "azimuth": np.linspace(0, 360, 5, endpoint=False),
                "range": np.linspace(0, 1000, 10),
            },
        )

        return DataTree.from_dict({"/": root_data, "sweep_0": ds})

    def create_dtree_with_dim(self, append_dim: str) -> DataTree:
        root_data = xr.Dataset(
            data_vars={
                "time_coverage_start": (
                    append_dim,
                    [pd.Timestamp("2023-01-01T00:00:00")],
                )
            },
            coords={append_dim: [pd.Timestamp("2023-01-01T00:00:00")]},
        )

        ds = xr.Dataset(
            data_vars={
                "reflectivity": (
                    [append_dim, "azimuth", "range"],
                    np.random.rand(1, 5, 10),
                )
            },
            coords={
                "azimuth": np.linspace(0, 360, 5, endpoint=False),
                "range": np.linspace(0, 1000, 10),
                append_dim: [pd.Timestamp("2023-01-01T00:00:00")],
            },
        )

        return DataTree.from_dict({"/": root_data, "sweep_0": ds})

    def test_exp_dim_adds_dimension_and_coordinates(self):
        dtree = self.create_dtree_without_dim()
        result = exp_dim(dtree, append_dim="vcp_time")

        for node in result.subtree:
            assert "vcp_time" in node.ds.dims
            assert "vcp_time" in node.ds.coords
            assert isinstance(node.ds["vcp_time"].values[0], np.datetime64)

    def test_ensure_dimension_adds_missing_dim(self):
        dtree = self.create_dtree_without_dim()
        result = ensure_dimension(dtree, append_dim="vcp_time")

        for node in result.subtree:
            assert "vcp_time" in node.ds.dims

    def test_ensure_dimension_skips_if_dim_exists(self):
        dtree = self.create_dtree_with_dim("vcp_time")
        result = ensure_dimension(dtree, append_dim="vcp_time")

        for node in result.subtree:
            assert "vcp_time" in node.ds.dims
        assert result == dtree


class TestSliceToVcpDimensions:
    """Test backward-compatible range dimension slicing for VCP configs."""

    @pytest.fixture
    def mock_vcp_config(self):
        """Mock VCP configuration with range dimensions."""
        return {
            "VCP-212": {
                "elevations": [0.5, 0.5, 0.9, 0.9],
                "dims": {
                    "azimuth": [720, 720, 720, 720],
                    "range": [1712, 1712, 1540, 1540],  # New 2025 config
                },
            },
            "VCP-32": {
                "elevations": [0.5, 1.5, 2.4],
                "dims": {
                    "azimuth": [720, 720, 720],
                    "range": [1832, 1832, 1832],
                },
            },
        }

    @pytest.fixture
    def dtree_old_data(self):
        """Create DataTree with old range dimensions (1832 bins)."""
        # Create mock datasets for sweeps with old range dimensions
        sweep_0 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1832)),
                "VEL": (["azimuth", "range"], np.random.rand(720, 1832)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1832),
            },
        )

        sweep_1 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1832)),
                "VEL": (["azimuth", "range"], np.random.rand(720, 1832)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1832),
            },
        )

        sweep_2 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1832)),
                "VEL": (["azimuth", "range"], np.random.rand(720, 1832)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1832),
            },
        )

        sweep_3 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1832)),
                "VEL": (["azimuth", "range"], np.random.rand(720, 1832)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1832),
            },
        )

        dtree = DataTree.from_dict(
            {
                "sweep_0": sweep_0,
                "sweep_1": sweep_1,
                "sweep_2": sweep_2,
                "sweep_3": sweep_3,
            }
        )
        dtree.attrs["scan_name"] = "VCP-212"
        return dtree

    @pytest.fixture
    def dtree_new_data(self):
        """Create DataTree with new range dimensions (1712 bins)."""
        # Create mock datasets for sweeps with new range dimensions
        sweep_0 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1712)),
                "VEL": (["azimuth", "range"], np.random.rand(720, 1712)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1712),
            },
        )

        sweep_1 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1712)),
                "VEL": (["azimuth", "range"], np.random.rand(720, 1712)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1712),
            },
        )

        dtree = DataTree.from_dict(
            {
                "sweep_0": sweep_0,
                "sweep_1": sweep_1,
            }
        )
        dtree.attrs["scan_name"] = "VCP-212"
        return dtree

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_slice_old_data_to_new_config(
        self, mock_template_manager, dtree_old_data, mock_vcp_config
    ):
        """Test slicing old data (1832 bins) to new config (1712 bins)."""
        mock_template_manager.return_value.config = mock_vcp_config

        result = slice_to_vcp_dimensions(
            dtree_old_data, vcp="VCP-212", vcp_config_file="vcp_nexrad.json"
        )

        # Check that range dimension was sliced for sweep_0 and sweep_1
        assert result["sweep_0"].dims["range"] == 1712
        assert result["sweep_1"].dims["range"] == 1712

        # Check that range dimension was sliced for sweep_2 and sweep_3
        assert result["sweep_2"].dims["range"] == 1540
        assert result["sweep_3"].dims["range"] == 1540

        # Verify data variables were sliced correctly
        assert result["sweep_0"]["DBZH"].shape == (720, 1712)
        assert result["sweep_0"]["VEL"].shape == (720, 1712)

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_new_data_unchanged(
        self, mock_template_manager, dtree_new_data, mock_vcp_config
    ):
        """Test that new data matching config is unchanged."""
        mock_template_manager.return_value.config = mock_vcp_config

        result = slice_to_vcp_dimensions(
            dtree_new_data, vcp="VCP-212", vcp_config_file="vcp_nexrad.json"
        )

        # Check that range dimension was NOT sliced (already matches config)
        assert result["sweep_0"].dims["range"] == 1712
        assert result["sweep_1"].dims["range"] == 1712

        # Verify data variables remain unchanged
        assert result["sweep_0"]["DBZH"].shape == (720, 1712)
        assert result["sweep_0"]["VEL"].shape == (720, 1712)

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_vcp_not_found_returns_unchanged(
        self, mock_template_manager, dtree_old_data
    ):
        """Test that missing VCP in config returns DataTree unchanged with warning."""
        mock_template_manager.return_value.config = {"VCP-32": {}}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = slice_to_vcp_dimensions(
                dtree_old_data, vcp="VCP-999", vcp_config_file="vcp_nexrad.json"
            )

            # Should emit warning about VCP not found
            assert len(w) == 1
            assert "VCP-999 not found in config file" in str(w[0].message)

        # DataTree should be unchanged
        assert result["sweep_0"].dims["range"] == 1832
        assert result["sweep_1"].dims["range"] == 1832

    def test_dynamic_scan_pattern_returns_unchanged(self, dtree_old_data):
        """Test that dynamic scan patterns (SAILS, MRLE, etc.) skip slicing."""
        # Test various dynamic scan patterns
        for vcp in ["SAILS", "MRLE", "AVSET", "MESO-SAILS×2", "SAILS×1"]:
            result = slice_to_vcp_dimensions(
                dtree_old_data, vcp=vcp, vcp_config_file="vcp_nexrad.json"
            )

            # DataTree should be unchanged (no slicing for dynamic scans)
            assert result["sweep_0"].dims["range"] == 1832
            assert result["sweep_1"].dims["range"] == 1832

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_missing_range_dims_in_config(self, mock_template_manager, dtree_old_data):
        """Test config without range dimensions returns DataTree unchanged."""
        # Config without 'dims' or 'range' key
        mock_template_manager.return_value.config = {
            "VCP-212": {
                "elevations": [0.5, 0.5, 0.9, 0.9],
                # Missing 'dims' key
            }
        }

        result = slice_to_vcp_dimensions(
            dtree_old_data, vcp="VCP-212", vcp_config_file="vcp_nexrad.json"
        )

        # DataTree should be unchanged
        assert result["sweep_0"].dims["range"] == 1832
        assert result["sweep_1"].dims["range"] == 1832

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_config_file_not_found(self, mock_template_manager, dtree_old_data):
        """Test handling of missing config file with warning."""
        mock_template_manager.side_effect = FileNotFoundError(
            "Config file not found: /path/to/missing.json"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = slice_to_vcp_dimensions(
                dtree_old_data,
                vcp="VCP-212",
                vcp_config_file="missing_config.json",
            )

            # Should emit warning about config file not found
            assert len(w) == 1
            assert "Could not load VCP config file" in str(w[0].message)

        # DataTree should be unchanged
        assert result["sweep_0"].dims["range"] == 1832
        assert result["sweep_1"].dims["range"] == 1832

    def test_empty_vcp_name(self, dtree_old_data):
        """Test empty VCP name returns DataTree unchanged."""
        result = slice_to_vcp_dimensions(
            dtree_old_data, vcp="", vcp_config_file="vcp_nexrad.json"
        )

        # DataTree should be unchanged
        assert result["sweep_0"].dims["range"] == 1832

    def test_none_vcp_name(self, dtree_old_data):
        """Test None VCP name returns DataTree unchanged."""
        result = slice_to_vcp_dimensions(
            dtree_old_data, vcp=None, vcp_config_file="vcp_nexrad.json"
        )

        # DataTree should be unchanged
        assert result["sweep_0"].dims["range"] == 1832

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_sweep_index_exceeds_config_length(
        self, mock_template_manager, dtree_old_data, mock_vcp_config
    ):
        """Test sweep index beyond config length is skipped."""
        # Add sweep_10 to DataTree but config only has 4 sweeps
        sweep_10 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1832)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1832),
            },
        )

        dtree_with_extra = DataTree.from_dict(
            {
                "sweep_0": dtree_old_data["sweep_0"].ds,
                "sweep_1": dtree_old_data["sweep_1"].ds,
                "sweep_10": sweep_10,
            }
        )

        mock_template_manager.return_value.config = mock_vcp_config

        result = slice_to_vcp_dimensions(
            dtree_with_extra, vcp="VCP-212", vcp_config_file="vcp_nexrad.json"
        )

        # sweep_0 and sweep_1 should be sliced
        assert result["sweep_0"].dims["range"] == 1712
        assert result["sweep_1"].dims["range"] == 1712

        # sweep_10 should be unchanged (index exceeds config)
        assert result["sweep_10"].dims["range"] == 1832

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_sweep_without_range_dimension(
        self, mock_template_manager, mock_vcp_config
    ):
        """Test sweep without range dimension is skipped."""
        # Create sweep without range dimension
        sweep_0 = xr.Dataset(
            {
                "azimuth_data": (["azimuth"], np.random.rand(720)),
            },
            coords={
                "azimuth": np.arange(720),
            },
        )

        dtree = DataTree.from_dict({"sweep_0": sweep_0})

        mock_template_manager.return_value.config = mock_vcp_config

        result = slice_to_vcp_dimensions(
            dtree, vcp="VCP-212", vcp_config_file="vcp_nexrad.json"
        )

        # Should return unchanged (no range dimension to slice)
        assert "range" not in result["sweep_0"].dims

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_preserves_datatree_attributes(
        self, mock_template_manager, dtree_old_data, mock_vcp_config
    ):
        """Test that DataTree attributes are preserved after slicing."""
        dtree_old_data.attrs["scan_name"] = "VCP-212"
        dtree_old_data.attrs["radar_name"] = "KVNX"
        dtree_old_data.attrs["custom_attr"] = "test_value"

        mock_template_manager.return_value.config = mock_vcp_config

        result = slice_to_vcp_dimensions(
            dtree_old_data, vcp="VCP-212", vcp_config_file="vcp_nexrad.json"
        )

        # Attributes should be preserved
        assert result.attrs["scan_name"] == "VCP-212"
        assert result.attrs["radar_name"] == "KVNX"
        assert result.attrs["custom_attr"] == "test_value"

    @patch("raw2zarr.transform.dimension.VcpTemplateManager")
    def test_mixed_range_dimensions(self, mock_template_manager, mock_vcp_config):
        """Test DataTree with mix of old and new range dimensions."""
        # sweep_0: old (1832), sweep_1: new (1712)
        sweep_0 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1832)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1832),
            },
        )

        sweep_1 = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], np.random.rand(720, 1712)),
            },
            coords={
                "azimuth": np.arange(720),
                "range": np.arange(1712),
            },
        )

        dtree = DataTree.from_dict({"sweep_0": sweep_0, "sweep_1": sweep_1})

        mock_template_manager.return_value.config = mock_vcp_config

        result = slice_to_vcp_dimensions(
            dtree, vcp="VCP-212", vcp_config_file="vcp_nexrad.json"
        )

        # sweep_0 should be sliced (1832 → 1712)
        assert result["sweep_0"].dims["range"] == 1712

        # sweep_1 should be unchanged (already 1712)
        assert result["sweep_1"].dims["range"] == 1712
