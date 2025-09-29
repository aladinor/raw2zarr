import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from raw2zarr.builder.builder_utils import (
    extract_file_metadata,
    extract_timestamp,
    get_icechunk_repo,
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


class TestLogProblematicFile:
    """Test the _log_problematic_file function."""

    def test_log_problematic_file_creates_output_file(self, tmp_path):
        """Test that _log_problematic_file creates output.txt file."""
        from raw2zarr.builder.builder_utils import _log_problematic_file

        # Change to temp directory to avoid affecting real output.txt
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Log a problematic file
            test_filepath = "s3://bucket/KVNX20110503_122230_V06.gz"
            test_error = "Template mismatch: PHIDP dimension error"

            _log_problematic_file(test_filepath, test_error)

            # Verify output.txt was created
            output_file = tmp_path / "output.txt"
            assert output_file.exists(), "output.txt should be created"

            # Verify content
            with open(output_file) as f:
                content = f.read()

            assert test_filepath in content
            assert test_error in content
            assert "SKIPPED:" in content

        finally:
            os.chdir(original_cwd)

    def test_log_problematic_file_appends_to_existing(self, tmp_path):
        """Test that _log_problematic_file appends to existing output.txt."""
        from raw2zarr.builder.builder_utils import _log_problematic_file

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Create initial output.txt
            output_file = tmp_path / "output.txt"
            with open(output_file, "w") as f:
                f.write("Initial content\n")

            # Log a problematic file
            test_filepath = "KVNX20110503_122230_V06.gz"
            test_error = "VCP-21 dimension error"

            _log_problematic_file(test_filepath, test_error)

            # Verify content was appended
            with open(output_file) as f:
                content = f.read()

            assert "Initial content" in content
            assert test_filepath in content
            assert test_error in content

            # Should have both old and new content
            lines = content.strip().split("\n")
            assert len(lines) >= 2

        finally:
            os.chdir(original_cwd)

    def test_log_problematic_file_encoding(self, tmp_path):
        """Test that _log_problematic_file handles UTF-8 encoding correctly."""
        from raw2zarr.builder.builder_utils import _log_problematic_file

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Test with special characters
            test_filepath = "файл_with_unicode.gz"  # Cyrillic characters
            test_error = "Error with unicode: ñoñó"

            _log_problematic_file(test_filepath, test_error)

            # Verify file can be read back correctly
            output_file = tmp_path / "output.txt"
            with open(output_file, encoding="utf-8") as f:
                content = f.read()

            assert test_filepath in content
            assert test_error in content

        finally:
            os.chdir(original_cwd)
