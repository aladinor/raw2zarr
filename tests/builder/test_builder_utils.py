import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from raw2zarr.builder.builder_utils import (
    get_icechunk_repo,
    extract_timestamp,
    extract_file_metadata,
)


class TestGetIcechunkRepo:
    """Test icechunk repository creation with manifest configuration."""

    def test_get_icechunk_repo_without_manifest_config(self, tmp_path):
        """Test repository creation without manifest configuration."""
        zarr_store = str(tmp_path / "test.zarr")

        repo = get_icechunk_repo(zarr_store, use_manifest_config=False)

        assert repo is not None
        assert os.path.exists(zarr_store)

    def test_get_icechunk_repo_with_manifest_config(self, tmp_path):
        """Test repository creation with manifest configuration."""
        zarr_store = str(tmp_path / "test_manifest.zarr")

        repo = get_icechunk_repo(zarr_store, use_manifest_config=True)

        assert repo is not None
        assert os.path.exists(zarr_store)

    def test_get_icechunk_repo_open_existing(self, tmp_path):
        """Test opening existing repository."""
        zarr_store = str(tmp_path / "existing.zarr")

        # Create repository first
        repo1 = get_icechunk_repo(zarr_store, use_manifest_config=True)
        assert repo1 is not None

        # Open existing repository
        repo2 = get_icechunk_repo(zarr_store, use_manifest_config=True)
        assert repo2 is not None

    @patch("raw2zarr.builder.builder_utils.icechunk")
    def test_manifest_config_parameters(self, mock_icechunk, tmp_path):
        """Test that manifest configuration is properly applied."""
        zarr_store = str(tmp_path / "manifest_test.zarr")

        # Mock icechunk components
        mock_storage = MagicMock()
        mock_repo = MagicMock()
        mock_icechunk.local_filesystem_storage.return_value = mock_storage
        mock_icechunk.Repository.create.return_value = mock_repo

        # Mock manifest config classes
        mock_split_config = MagicMock()
        mock_preload_config = MagicMock()
        mock_repo_config = MagicMock()

        mock_icechunk.ManifestSplittingConfig.from_dict.return_value = mock_split_config
        mock_icechunk.ManifestPreloadConfig.return_value = mock_preload_config
        mock_icechunk.RepositoryConfig.return_value = mock_repo_config

        # Call function
        get_icechunk_repo(zarr_store, use_manifest_config=True)

        # Verify manifest configuration was created
        mock_icechunk.ManifestSplittingConfig.from_dict.assert_called_once()
        mock_icechunk.ManifestPreloadConfig.assert_called_once()
        mock_icechunk.RepositoryConfig.assert_called_once()

        # Verify repository was created with config
        mock_icechunk.Repository.create.assert_called_once_with(
            mock_storage, config=mock_repo_config
        )


class TestExtractTimestamp:
    """Test timestamp extraction from radar filenames."""

    def test_extract_timestamp_nexrad_format(self):
        """Test NEXRAD filename format YYYYMMDD_HHMMSS."""
        filename = "KVNX20110520_000023_V06"
        expected = pd.Timestamp("2011-05-20 00:00:23")

        result = extract_timestamp(filename)

        assert result == expected

    def test_extract_timestamp_legacy_format(self):
        """Test legacy filename format with 6-digit date."""
        filename = "ABC110520000023"
        expected = pd.Timestamp("2011-05-20 00:00:23")

        result = extract_timestamp(filename)

        assert result == expected

    def test_extract_timestamp_invalid_format(self):
        """Test that invalid filename raises ValueError."""
        filename = "invalid_filename_format"

        with pytest.raises(ValueError, match="Could not parse timestamp"):
            extract_timestamp(filename)

    def test_extract_timestamp_with_path(self):
        """Test timestamp extraction from full file path."""
        filepath = "/path/to/data/KVNX20110520_000023_V06.gz"
        expected = pd.Timestamp("2011-05-20 00:00:23")

        result = extract_timestamp(filepath)

        assert result == expected


class TestExtractFileMetadata:
    """Test combined timestamp and VCP extraction."""

    @patch("raw2zarr.builder.builder_utils.NEXRADLevel2File")
    @patch("raw2zarr.builder.builder_utils.normalize_input_for_xradar")
    def test_extract_file_metadata_nexrad(self, mock_normalize, mock_nexrad_file):
        """Test metadata extraction from NEXRAD file."""
        filename = "KVNX20110520_000023_V06"
        mock_normalize.return_value = filename

        # Mock VCP data
        mock_file_instance = MagicMock()
        mock_file_instance.get_msg_5_data.return_value = {"pattern_number": 12}
        mock_nexrad_file.return_value = mock_file_instance

        timestamp, vcp = extract_file_metadata(filename, "nexradlevel2")

        assert timestamp == pd.Timestamp("2011-05-20 00:00:23")
        assert vcp == 12
        mock_normalize.assert_called_once_with(filename)
        mock_nexrad_file.assert_called_once_with(filename)

    def test_extract_file_metadata_unsupported_engine(self):
        """Test that unsupported engine raises ValueError."""
        filename = "KVNX20110520_000023_V06"  # Valid filename format

        with pytest.raises(ValueError, match="Engine not supported"):
            extract_file_metadata(filename, "unsupported_engine")

    @patch("raw2zarr.builder.builder_utils.NEXRADLevel2File")
    @patch("raw2zarr.builder.builder_utils.normalize_input_for_xradar")
    def test_extract_file_metadata_integration(self, mock_normalize, mock_nexrad_file):
        """Test that both timestamp and VCP are extracted correctly."""
        filename = "KVNX20110520_120000_V06"
        mock_normalize.return_value = filename

        # Mock VCP data
        mock_file_instance = MagicMock()
        mock_file_instance.get_msg_5_data.return_value = {"pattern_number": 212}
        mock_nexrad_file.return_value = mock_file_instance

        timestamp, vcp = extract_file_metadata(filename)

        # Verify both values are extracted correctly
        assert isinstance(timestamp, pd.Timestamp)
        assert timestamp == pd.Timestamp("2011-05-20 12:00:00")
        assert isinstance(vcp, int)
        assert vcp == 212


@pytest.fixture
def sample_radar_filename():
    """Sample radar filename for testing."""
    return "KVNX20110520_000023_V06"


@pytest.fixture
def temp_zarr_store():
    """Temporary zarr store for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield os.path.join(tmp_dir, "test.zarr")
