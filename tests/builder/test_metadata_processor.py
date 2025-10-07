"""Tests for metadata processing workflow."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from raw2zarr.builder.metadata_processor import (
    MetadataProcessingResult,
    flatten_metadata_results,
    process_metadata_and_create_vcp_mapping,
)


class TestFlattenMetadataResults:
    """Test flattening of nested metadata results."""

    def test_flatten_standard_scans(self):
        """Test flattening list of single tuples (standard scans)."""
        nested = [
            [
                (
                    0,
                    "file1.h5",
                    pd.Timestamp("2023-01-01 12:00:00"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
            [
                (
                    1,
                    "file2.h5",
                    pd.Timestamp("2023-01-01 12:05:00"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
        ]

        result = flatten_metadata_results(nested)

        assert len(result) == 2
        assert result[0][1] == "file1.h5"
        assert result[1][1] == "file2.h5"

    def test_flatten_dynamic_scans_with_multiple_slices(self):
        """Test flattening MESO-SAILS with multiple slices per file."""
        nested = [
            [
                (
                    0,
                    "file1.h5",
                    pd.Timestamp("2023-01-01 12:00:00"),
                    "VCP-212",
                    0,
                    [0, 1, 2],
                    "MESO-SAILS×2",
                    [0.5, 0.5, 0.9],
                ),
                (
                    0,
                    "file1.h5",
                    pd.Timestamp("2023-01-01 12:02:00"),
                    "VCP-212",
                    1,
                    [6, 7, 8],
                    "MESO-SAILS×2",
                    [0.5, 0.5, 0.9],
                ),
            ],
        ]

        result = flatten_metadata_results(nested)

        # Should flatten to 2 entries (2 temporal slices)
        assert len(result) == 2
        assert result[0][4] == 0  # slice_id of first entry
        assert result[1][4] == 1  # slice_id of second entry

    def test_flatten_mixed_standard_and_dynamic(self):
        """Test flattening mix of standard and dynamic scans."""
        nested = [
            # Standard scan - single tuple in list
            [
                (
                    0,
                    "file1.h5",
                    pd.Timestamp("2023-01-01 12:00:00"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
            # Dynamic scan - multiple tuples in list
            [
                (
                    1,
                    "file2.h5",
                    pd.Timestamp("2023-01-01 12:05:00"),
                    "VCP-212",
                    0,
                    [0, 1],
                    "SAILS",
                    [0.5, 0.5],
                ),
                (
                    1,
                    "file2.h5",
                    pd.Timestamp("2023-01-01 12:07:00"),
                    "VCP-212",
                    1,
                    [6, 7],
                    "SAILS",
                    [0.5, 0.5],
                ),
            ],
        ]

        result = flatten_metadata_results(nested)

        assert len(result) == 3  # 1 standard + 2 slices


class TestProcessMetadataWithCorruption:
    """Test metadata processing with corrupted file filtering."""

    @pytest.fixture
    def mock_client(self):
        """Mock Dask distributed client."""
        return MagicMock()

    def test_filters_corrupted_files(self, mock_client, tmp_path):
        """Test that corrupted files are filtered out."""
        # Mock metadata extraction results with one corrupted file
        metadata_results = [
            [
                (
                    0,
                    "file1.h5",
                    pd.Timestamp("2023-01-01 12:00:00"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
            [
                (
                    1,
                    "file2.h5",
                    "ERROR",
                    "Corrupted: 12/14 sweeps have data",
                    0,
                    [],
                    "CORRUPTED",
                    [],
                )
            ],
            [
                (
                    2,
                    "file3.h5",
                    pd.Timestamp("2023-01-01 12:10:00"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
        ]

        mock_client.map.return_value = []
        mock_client.gather.return_value = metadata_results

        log_file = tmp_path / "test_output.txt"

        result = process_metadata_and_create_vcp_mapping(
            client=mock_client,
            radar_files=["file1.h5", "file2.h5", "file3.h5"],
            engine="nexradlevel2",
            log_file=str(log_file),
        )

        # Should have 2 valid files (file1 and file3)
        assert result is not None
        assert result.total_valid_files == 2
        assert result.problematic_count == 1

        # Verify corrupted file was logged
        assert log_file.exists()
        with open(log_file) as f:
            content = f.read()
        assert "file2.h5" in content
        assert "Corrupted" in content

    def test_all_corrupted_returns_none(self, mock_client):
        """Test that all corrupted files returns None."""
        metadata_results = [
            [
                (
                    0,
                    "file1.h5",
                    "ERROR",
                    "Corrupted: 10/14 sweeps",
                    0,
                    [],
                    "CORRUPTED",
                    [],
                )
            ],
            [
                (
                    1,
                    "file2.h5",
                    "ERROR",
                    "Corrupted: 11/14 sweeps",
                    0,
                    [],
                    "CORRUPTED",
                    [],
                )
            ],
        ]

        mock_client.map.return_value = []
        mock_client.gather.return_value = metadata_results

        result = process_metadata_and_create_vcp_mapping(
            client=mock_client,
            radar_files=["file1.h5", "file2.h5"],
            engine="nexradlevel2",
        )

        # Should return None when all files are corrupted
        assert result is None

    def test_skip_vcps_filters_correctly(self, mock_client, tmp_path):
        """Test that specified VCPs are skipped."""
        metadata_results = [
            [
                (
                    0,
                    "file1.h5",
                    pd.Timestamp("2023-01-01"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
            [
                (
                    1,
                    "file2.h5",
                    pd.Timestamp("2023-01-01"),
                    "VCP-31",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
            [
                (
                    2,
                    "file3.h5",
                    pd.Timestamp("2023-01-01"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
        ]

        mock_client.map.return_value = []
        mock_client.gather.return_value = metadata_results

        log_file = tmp_path / "test_output.txt"

        result = process_metadata_and_create_vcp_mapping(
            client=mock_client,
            radar_files=["file1.h5", "file2.h5", "file3.h5"],
            engine="nexradlevel2",
            skip_vcps=["VCP-31"],
            log_file=str(log_file),
        )

        # Should have 2 valid files (VCP-12 only)
        assert result is not None
        assert result.total_valid_files == 2
        assert result.skipped_vcp_count == 1

        # Verify skipped VCP was logged
        content = log_file.read_text()
        assert "file2.h5" in content
        assert "VCP-31" in content

    def test_vcp_time_mapping_created(self, mock_client):
        """Test that VCP time mapping is created correctly."""
        metadata_results = [
            [
                (
                    0,
                    "file1.h5",
                    pd.Timestamp("2023-01-01 12:00:00"),
                    "VCP-12",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
            [
                (
                    1,
                    "file2.h5",
                    pd.Timestamp("2023-01-01 12:05:00"),
                    "VCP-212",
                    0,
                    [0, 1],
                    "STANDARD",
                    None,
                )
            ],
        ]

        mock_client.map.return_value = []
        mock_client.gather.return_value = metadata_results

        result = process_metadata_and_create_vcp_mapping(
            client=mock_client,
            radar_files=["file1.h5", "file2.h5"],
            engine="nexradlevel2",
        )

        # Verify VCP time mapping structure
        assert result is not None
        assert "VCP-12" in result.vcp_names
        assert "VCP-212" in result.vcp_names
        assert len(result.vcp_names) == 2


class TestMetadataProcessingResult:
    """Test MetadataProcessingResult data class."""

    def test_basic_properties(self):
        """Test basic properties of MetadataProcessingResult."""
        vcp_mapping = {
            "VCP-12": {"file_count": 5, "timestamps": []},
            "VCP-212": {"file_count": 3, "timestamps": []},
        }

        result = MetadataProcessingResult(
            vcp_time_mapping=vcp_mapping,
            valid_files=[(0, "f1"), (1, "f2")],
            problematic_count=2,
            skipped_vcp_count=1,
        )

        assert result.total_valid_files == 8  # 5 + 3
        assert result.problematic_count == 2
        assert result.skipped_vcp_count == 1
        assert len(result.vcp_names) == 2
        assert "VCP-12" in result.vcp_names
        assert "VCP-212" in result.vcp_names
