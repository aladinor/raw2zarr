"""Tests for VCP utility functions."""

import pandas as pd
import pytest
from xarray import DataTree

from raw2zarr.templates.vcp_utils import (
    create_vcp_time_mapping_with_slices,
    map_sweeps_to_vcp_indices,
)


class TestMapSweepsToVcpIndices:
    """Test VCP sweep index mapping for dynamic scans."""

    def test_returns_unchanged_when_elevation_angles_none(self):
        """Standard scans (no elevation angles) return unchanged."""
        # Create minimal DataTree
        dtree = DataTree(name="root")

        result = map_sweeps_to_vcp_indices(
            data_tree=dtree,
            vcp="VCP-212",
            sweep_indices=None,
            elevation_angles=None,
        )

        # Should return unchanged for standard scans
        assert result is dtree

    def test_returns_unchanged_when_vcp_not_in_config(self):
        """Unknown VCP patterns return unchanged."""
        from xarray import Dataset

        dtree = DataTree(Dataset())

        result = map_sweeps_to_vcp_indices(
            data_tree=dtree,
            vcp="VCP-999",  # Non-existent VCP
            sweep_indices=[0, 1, 2],
            elevation_angles=[0.5, 0.9, 1.3],
        )

        # Should return unchanged when VCP not found
        assert result is dtree

    @pytest.mark.integration
    def test_standard_vcp_file_loads_correctly(self, nexrad_standard_vcp_file):
        """Test with real standard VCP file (integration test)."""
        from raw2zarr.io.load import load_radar_data

        # Load the standard VCP file
        dtree = load_radar_data(nexrad_standard_vcp_file, engine="nexradlevel2")

        # Standard scan - no mapping needed (elevation_angles=None)
        result = map_sweeps_to_vcp_indices(
            data_tree=dtree,
            vcp="VCP-212",
            sweep_indices=None,
            elevation_angles=None,
        )

        # Should return unchanged
        assert result is dtree

    @pytest.mark.integration
    def test_dynamic_vcp_mapping_with_real_file(self, nexrad_dynamic_vcp_file):
        """Test MESO-SAILS sweep mapping with real file (integration test)."""
        from raw2zarr.builder.builder_utils import extract_single_metadata
        from raw2zarr.io.load import load_radar_data

        # Get metadata for the dynamic file
        metadata = extract_single_metadata(
            (0, nexrad_dynamic_vcp_file), engine="nexradlevel2"
        )

        # MESO-SAILS should have multiple temporal slices
        assert len(metadata) > 1, "Dynamic scan should have multiple temporal slices"

        # Load the file
        dtree = load_radar_data(nexrad_dynamic_vcp_file, engine="nexradlevel2")

        # Test mapping for first temporal slice
        first_slice = metadata[0]
        sweep_indices = first_slice[5]  # sweep_indices field
        elevation_angles = first_slice[7]  # elevation_angles field
        vcp = first_slice[3]  # VCP name

        result = map_sweeps_to_vcp_indices(
            data_tree=dtree,
            vcp=vcp,
            sweep_indices=sweep_indices,
            elevation_angles=elevation_angles,
        )

        # Verify result is a DataTree
        assert isinstance(result, DataTree)

        # Should have remapped sweeps (sweep_0, sweep_1, etc.)
        assert len(result.children) > 0 or "/" in result.groups


class TestCreateVcpTimeMappingWithSlices:
    """Test VCP time mapping creation with temporal slices."""

    def test_single_vcp_no_slices(self):
        """Test mapping for single VCP with standard scans."""
        metadata_results = [
            (
                pd.Timestamp("2023-01-01 12:00:00"),
                "VCP-12",
                0,
                [0, 1, 2],
                "STANDARD",
                None,
            ),
            (
                pd.Timestamp("2023-01-01 12:05:00"),
                "VCP-12",
                0,
                [0, 1, 2],
                "STANDARD",
                None,
            ),
        ]
        valid_files = [(0, "file1.h5"), (1, "file2.h5")]

        result = create_vcp_time_mapping_with_slices(metadata_results, valid_files)

        assert "VCP-12" in result
        assert result["VCP-12"]["file_count"] == 2
        assert len(result["VCP-12"]["timestamps"]) == 2
        assert result["VCP-12"]["time_range"] == (0, 2)

    def test_multiple_vcps(self):
        """Test mapping with multiple VCP patterns."""
        metadata_results = [
            (
                pd.Timestamp("2023-01-01 12:00:00"),
                "VCP-212",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
            (
                pd.Timestamp("2023-01-01 12:05:00"),
                "VCP-35",
                0,
                [0, 1, 2],
                "STANDARD",
                None,
            ),
            (
                pd.Timestamp("2023-01-01 12:10:00"),
                "VCP-212",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
        ]
        valid_files = [(0, "file1.h5"), (1, "file2.h5"), (2, "file3.h5")]

        result = create_vcp_time_mapping_with_slices(metadata_results, valid_files)

        # Should have 2 different VCPs
        assert len(result) == 2
        assert "VCP-212" in result
        assert "VCP-35" in result

        # VCP-212 has 2 files (index 0 and 2)
        assert result["VCP-212"]["file_count"] == 2

        # VCP-35 has 1 file (index 1)
        assert result["VCP-35"]["file_count"] == 1

    def test_temporal_slices_from_same_file(self):
        """Test MESO-SAILS temporal slices from same file treated as separate entries."""
        metadata_results = [
            (
                pd.Timestamp("2023-01-01 12:00:00"),
                "VCP-212",
                0,
                [0, 1, 2],
                "MESO-SAILS×2",
                [0.5, 0.5, 0.9],
            ),
            (
                pd.Timestamp("2023-01-01 12:02:00"),
                "VCP-212",
                1,
                [6, 7, 8],
                "MESO-SAILS×2",
                [0.5, 0.5, 0.9],
            ),
        ]
        # Same file, different temporal slices
        valid_files = [(0, "file1.h5"), (1, "file1.h5")]

        result = create_vcp_time_mapping_with_slices(metadata_results, valid_files)

        # Two temporal slices = 2 "virtual files"
        assert result["VCP-212"]["file_count"] == 2

        # Verify both slices are tracked
        assert len(result["VCP-212"]["files"]) == 2
        assert result["VCP-212"]["files"][0]["slice_id"] == 0
        assert result["VCP-212"]["files"][1]["slice_id"] == 1

    def test_time_range_assignment(self):
        """Test monotonic time block assignment across VCPs."""
        metadata_results = [
            (
                pd.Timestamp("2023-01-01 12:00:00"),
                "VCP-212",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
            (
                pd.Timestamp("2023-01-01 12:05:00"),
                "VCP-212",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
            (
                pd.Timestamp("2023-01-01 12:10:00"),
                "VCP-35",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
        ]
        valid_files = [(0, "f1"), (1, "f2"), (2, "f3")]

        result = create_vcp_time_mapping_with_slices(metadata_results, valid_files)

        # VCP-212 should have time_range (0, 2) - indices 0 and 1
        assert result["VCP-212"]["time_range"] == (0, 2)

        # VCP-35 should have time_range (2, 3) - index 2
        assert result["VCP-35"]["time_range"] == (2, 3)

    def test_timestamps_sorted_by_time(self):
        """Test that timestamps are sorted chronologically within each VCP."""
        # Out of order timestamps
        metadata_results = [
            (
                pd.Timestamp("2023-01-01 12:10:00"),
                "VCP-12",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
            (
                pd.Timestamp("2023-01-01 12:00:00"),
                "VCP-12",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
            (
                pd.Timestamp("2023-01-01 12:05:00"),
                "VCP-12",
                0,
                [0, 1],
                "STANDARD",
                None,
            ),
        ]
        valid_files = [(0, "f1"), (1, "f2"), (2, "f3")]

        result = create_vcp_time_mapping_with_slices(metadata_results, valid_files)

        # Timestamps should be sorted
        timestamps = result["VCP-12"]["timestamps"]
        assert timestamps[0] == pd.Timestamp("2023-01-01 12:00:00")
        assert timestamps[1] == pd.Timestamp("2023-01-01 12:05:00")
        assert timestamps[2] == pd.Timestamp("2023-01-01 12:10:00")

    def test_files_metadata_preserved(self):
        """Test that all file metadata fields are preserved."""
        metadata_results = [
            (
                pd.Timestamp("2023-01-01 12:00:00"),
                "VCP-212",
                0,
                [0, 1, 2, 3],
                "MESO-SAILS×2",
                [0.5, 0.5, 0.9, 0.9],
            ),
        ]
        valid_files = [(0, "s3://bucket/file1.h5")]

        result = create_vcp_time_mapping_with_slices(metadata_results, valid_files)

        # Check that all fields are preserved
        file_entry = result["VCP-212"]["files"][0]
        assert file_entry["filepath"] == "s3://bucket/file1.h5"
        assert file_entry["timestamp"] == pd.Timestamp("2023-01-01 12:00:00")
        assert file_entry["slice_id"] == 0
        assert file_entry["sweep_indices"] == [0, 1, 2, 3]
        assert file_entry["scan_type"] == "MESO-SAILS×2"
        assert file_entry["elevation_angles"] == [0.5, 0.5, 0.9, 0.9]
        assert file_entry["file_index"] == 0
        assert file_entry["original_position"] == 0
