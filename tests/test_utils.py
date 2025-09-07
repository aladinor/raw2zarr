#!/usr/bin/env python
"""
Comprehensive test suite for raw2zarr.utils module.

Tests cover all main functions and helper functions with various scenarios
including edge cases, error conditions, and mocking external dependencies.
"""
import json
import os
import time
from datetime import datetime
from unittest.mock import Mock, mock_open, patch

import pytest

from raw2zarr.utils import (
    _get_files_for_date,
    _parse_nexrad_filename,
    create_query,
    list_nexrad_files,
    load_vcp_samples,
    make_dir,
    timer_func,
)


class TestTimerFunc:
    """Test suite for timer_func decorator."""

    def test_timer_func_basic_functionality(self, capsys):
        """Test basic timing functionality."""

        @timer_func
        def sample_function():
            time.sleep(0.01)  # Small delay for timing
            return "test_result"

        result = sample_function()
        captured = capsys.readouterr()

        assert result == "test_result"
        assert "executed in" in captured.out
        assert "sample_function" in captured.out

    def test_timer_func_preserves_function_metadata(self):
        """Test that @timer_func preserves function metadata with functools.wraps."""

        @timer_func
        def documented_function():
            """This function has documentation."""
            return 42

        # Check that functools.wraps preserved the original function metadata
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function has documentation."

    def test_timer_func_with_arguments(self, capsys):
        """Test timer_func with function arguments."""

        @timer_func
        def function_with_args(a, b, keyword=None):
            return a + b if keyword is None else a + b + keyword

        result = function_with_args(1, 2, keyword=3)
        captured = capsys.readouterr()

        assert result == 6
        assert "executed in" in captured.out

    def test_timer_func_with_exception(self, capsys):
        """Test timer_func behavior when decorated function raises exception."""

        @timer_func
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        captured = capsys.readouterr()
        # Should not print timing info when function fails
        assert captured.out == ""


class TestMakeDir:
    """Test suite for make_dir function."""

    def test_make_dir_creates_directory(self, tmp_path):
        """Test that make_dir creates a new directory."""
        new_dir = tmp_path / "test_directory"
        assert not new_dir.exists()

        make_dir(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_make_dir_with_existing_directory(self, tmp_path):
        """Test that make_dir doesn't fail with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        # Should not raise an exception
        make_dir(str(existing_dir))
        assert existing_dir.exists()

    def test_make_dir_creates_nested_directories(self, tmp_path):
        """Test that make_dir creates nested directory structure."""
        nested_path = tmp_path / "level1" / "level2" / "level3"
        assert not nested_path.exists()

        make_dir(str(nested_path))

        assert nested_path.exists()
        assert nested_path.is_dir()

    def test_make_dir_with_empty_string(self):
        """Test make_dir behavior with empty string."""
        # Empty string causes FileNotFoundError with os.makedirs
        with pytest.raises(FileNotFoundError):
            make_dir("")

    @patch("os.makedirs")
    def test_make_dir_calls_os_makedirs_correctly(self, mock_makedirs):
        """Test that make_dir calls os.makedirs with correct parameters."""
        test_path = "/test/path"
        make_dir(test_path)

        mock_makedirs.assert_called_once_with(test_path, exist_ok=True)


class TestCreateQuery:
    """Test suite for create_query function."""

    def test_create_query_basic_functionality(self):
        """Test basic query creation."""
        site = "guaviare"
        date = datetime(2022, 6, 1, 10, 30, 15)

        result = create_query(site, date)
        expected = "l2_data/2022/06/01/GUAVIARE/GUA220601"

        assert result == expected

    def test_create_query_with_prod_parameter(self):
        """Test create_query with prod parameter (currently not used)."""
        site = "test_site"
        date = datetime(2023, 12, 25, 15, 45)

        result = create_query(site, date, prod="custom")
        expected = "l2_data/2023/12/25/TEST_SITE/TES231225"

        assert result == expected

    def test_create_query_case_handling(self):
        """Test that site names are properly converted to uppercase."""
        site = "lowercase_site"
        date = datetime(2021, 1, 1)

        result = create_query(site, date)
        expected = "l2_data/2021/01/01/LOWERCASE_SITE/LOW210101"

        assert result == expected

    def test_create_query_with_invalid_site_empty(self):
        """Test create_query raises ValueError for empty site."""
        date = datetime(2022, 6, 1)

        with pytest.raises(ValueError, match="site must be a non-empty string"):
            create_query("", date)

    def test_create_query_with_invalid_site_none(self):
        """Test create_query raises ValueError for None site."""
        date = datetime(2022, 6, 1)

        with pytest.raises(ValueError, match="site must be a non-empty string"):
            create_query(None, date)

    def test_create_query_with_invalid_site_not_string(self):
        """Test create_query raises ValueError for non-string site."""
        date = datetime(2022, 6, 1)

        with pytest.raises(ValueError, match="site must be a non-empty string"):
            create_query(123, date)

    def test_create_query_with_invalid_date_not_datetime(self):
        """Test create_query raises TypeError for non-datetime date."""
        site = "test"

        with pytest.raises(TypeError, match="date must be a datetime object"):
            create_query(site, "2022-06-01")

    def test_create_query_with_invalid_date_none(self):
        """Test create_query raises TypeError for None date."""
        site = "test"

        with pytest.raises(TypeError, match="date must be a datetime object"):
            create_query(site, None)


class TestLoadVcpSamples:
    """Test suite for load_vcp_samples function."""

    def test_load_vcp_samples_success(self, tmp_path):
        """Test successful loading of VCP samples."""
        # Create test JSON file
        test_data = {
            "VCP-11": ["s3://bucket/file1.gz", "s3://bucket/file2.gz"],
            "VCP-12": ["s3://bucket/file3.gz", "s3://bucket/file4.gz"],
        }
        test_file = tmp_path / "test_vcp_samples.json"
        test_file.write_text(json.dumps(test_data))

        result = load_vcp_samples(str(test_file))

        assert result == test_data
        assert "VCP-11" in result
        assert len(result["VCP-11"]) == 2

    def test_load_vcp_samples_file_not_found(self):
        """Test load_vcp_samples raises FileNotFoundError for missing file."""
        non_existent_file = "/non/existent/path.json"

        with pytest.raises(
            FileNotFoundError, match=f"VCP samples file not found: {non_existent_file}"
        ):
            load_vcp_samples(non_existent_file)

    def test_load_vcp_samples_invalid_json(self, tmp_path):
        """Test load_vcp_samples raises JSONDecodeError for invalid JSON."""
        # Create file with invalid JSON
        test_file = tmp_path / "invalid.json"
        test_file.write_text("{ invalid json content")

        with pytest.raises(
            json.JSONDecodeError, match="Error parsing VCP samples JSON"
        ):
            load_vcp_samples(str(test_file))

    def test_load_vcp_samples_empty_json(self, tmp_path):
        """Test load_vcp_samples with empty JSON object."""
        test_file = tmp_path / "empty.json"
        test_file.write_text("{}")

        result = load_vcp_samples(str(test_file))

        assert result == {}

    @patch("builtins.open", mock_open(read_data='{"VCP-11": ["file1", "file2"]}'))
    def test_load_vcp_samples_mocked_file_operations(self):
        """Test load_vcp_samples with mocked file operations."""
        result = load_vcp_samples("dummy_path.json")

        expected = {"VCP-11": ["file1", "file2"]}
        assert result == expected


class TestParseNexradFilename:
    """Test suite for _parse_nexrad_filename helper function."""

    def test_parse_nexrad_filename_success(self):
        """Test successful parsing of NEXRAD filename."""
        filename = "KVNX20110520_093548_V06"
        radar = "KVNX"

        result = _parse_nexrad_filename(filename, radar)
        expected = datetime(2011, 5, 20, 9, 35, 48)

        assert result == expected

    def test_parse_nexrad_filename_with_gz_extension(self):
        """Test parsing NEXRAD filename with .gz extension."""
        filename = "KTLX20130520_194528_V06.gz"
        radar = "KTLX"

        result = _parse_nexrad_filename(filename, radar)
        expected = datetime(2013, 5, 20, 19, 45, 28)

        assert result == expected

    def test_parse_nexrad_filename_wrong_radar(self):
        """Test parsing with wrong radar code returns None."""
        filename = "KVNX20110520_093548_V06"
        radar = "KTLX"  # Wrong radar

        result = _parse_nexrad_filename(filename, radar)

        assert result is None

    def test_parse_nexrad_filename_invalid_format(self):
        """Test parsing invalid filename format returns None."""
        filename = "invalid_filename_format"
        radar = "KVNX"

        result = _parse_nexrad_filename(filename, radar)

        assert result is None

    def test_parse_nexrad_filename_invalid_date(self):
        """Test parsing filename with invalid date returns None."""
        filename = "KVNX20119999_999999_V06"  # Invalid date/time
        radar = "KVNX"

        result = _parse_nexrad_filename(filename, radar)

        assert result is None

    def test_parse_nexrad_filename_empty_filename(self):
        """Test parsing empty filename returns None."""
        result = _parse_nexrad_filename("", "KVNX")

        assert result is None


class TestGetFilesForDate:
    """Test suite for _get_files_for_date helper function."""

    @patch("fsspec.AbstractFileSystem")
    def test_get_files_for_date_success(self, mock_fs_class):
        """Test successful file retrieval for a date."""
        # Setup mock filesystem
        mock_fs = Mock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.return_value = [
            "noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",
            "noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_100147_V06.gz",
        ]

        radar = "KVNX"
        date = datetime(2011, 5, 20)
        start_dt = datetime(2011, 5, 20, 9, 0)
        end_dt = datetime(2011, 5, 20, 11, 0)

        result = _get_files_for_date(mock_fs, radar, date, start_dt, end_dt)

        expected = [
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_100147_V06.gz",
        ]
        assert result == expected

    @patch("fsspec.AbstractFileSystem")
    def test_get_files_for_date_time_filtering(self, mock_fs_class):
        """Test that files are properly filtered by time range."""
        mock_fs = Mock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.return_value = [
            "noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_083000_V06",  # Before range
            "noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",  # In range
            "noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_120000_V06",  # After range
        ]

        radar = "KVNX"
        date = datetime(2011, 5, 20)
        start_dt = datetime(2011, 5, 20, 9, 0)
        end_dt = datetime(2011, 5, 20, 11, 0)

        result = _get_files_for_date(mock_fs, radar, date, start_dt, end_dt)

        expected = ["s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06"]
        assert result == expected

    @patch("fsspec.AbstractFileSystem")
    def test_get_files_for_date_with_exceptions(self, mock_fs_class):
        """Test that exceptions are handled gracefully."""
        mock_fs = Mock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.side_effect = FileNotFoundError("S3 access error")

        radar = "KVNX"
        date = datetime(2011, 5, 20)
        start_dt = datetime(2011, 5, 20, 9, 0)
        end_dt = datetime(2011, 5, 20, 11, 0)

        result = _get_files_for_date(mock_fs, radar, date, start_dt, end_dt)

        assert result == []  # Should return empty list on error

    @patch("fsspec.AbstractFileSystem")
    def test_get_files_for_date_invalid_filenames(self, mock_fs_class):
        """Test handling of invalid filenames."""
        mock_fs = Mock()
        mock_fs_class.return_value = mock_fs
        mock_fs.glob.return_value = [
            "noaa-nexrad-level2/2011/05/20/KVNX/invalid_filename",
            "noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",
        ]

        radar = "KVNX"
        date = datetime(2011, 5, 20)
        start_dt = datetime(2011, 5, 20, 9, 0)
        end_dt = datetime(2011, 5, 20, 11, 0)

        result = _get_files_for_date(mock_fs, radar, date, start_dt, end_dt)

        # Should only return valid filename
        expected = ["s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06"]
        assert result == expected


class TestListNexradFiles:
    """Test suite for list_nexrad_files function."""

    @patch("raw2zarr.utils.fsspec.filesystem")
    @patch("raw2zarr.utils._get_files_for_date")
    def test_list_nexrad_files_success(self, mock_get_files, mock_filesystem):
        """Test successful NEXRAD file listing."""
        mock_fs = Mock()
        mock_filesystem.return_value = mock_fs
        mock_get_files.return_value = [
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_100147_V06",
        ]

        result = list_nexrad_files(
            radar="KVNX", start_time="2011-05-20 09:00", end_time="2011-05-20 11:00"
        )

        expected = [
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_100147_V06",
        ]
        assert result == expected
        mock_filesystem.assert_called_once_with("s3", anon=True)

    @patch("raw2zarr.utils.fsspec.filesystem")
    @patch("raw2zarr.utils._get_files_for_date")
    def test_list_nexrad_files_multiple_days(self, mock_get_files, mock_filesystem):
        """Test NEXRAD file listing across multiple days."""
        mock_fs = Mock()
        mock_filesystem.return_value = mock_fs

        # Mock returns different files for different dates
        def mock_get_files_side_effect(fs, radar, date, start_dt, end_dt):
            if date.day == 20:
                return ["s3://noaa-nexrad-level2/2011/05/20/KVNX/file1"]
            elif date.day == 21:
                return ["s3://noaa-nexrad-level2/2011/05/21/KVNX/file2"]
            return []

        mock_get_files.side_effect = mock_get_files_side_effect

        result = list_nexrad_files(
            radar="KVNX", start_time="2011-05-20 23:00", end_time="2011-05-21 01:00"
        )

        expected = [
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/file1",
            "s3://noaa-nexrad-level2/2011/05/21/KVNX/file2",
        ]
        assert result == expected
        assert mock_get_files.call_count == 2  # Called for both dates

    def test_list_nexrad_files_invalid_time_format(self):
        """Test list_nexrad_files raises ValueError for invalid time format."""
        with pytest.raises(ValueError, match="Invalid time format"):
            list_nexrad_files(
                radar="KVNX",
                start_time="invalid-time-format",
                end_time="2011-05-20 11:00",
            )

    def test_list_nexrad_files_start_after_end(self):
        """Test list_nexrad_files raises ValueError when start_time > end_time."""
        with pytest.raises(
            ValueError, match="start_time must be before or equal to end_time"
        ):
            list_nexrad_files(
                radar="KVNX", start_time="2011-05-20 11:00", end_time="2011-05-20 09:00"
            )

    def test_list_nexrad_files_invalid_radar_empty(self):
        """Test list_nexrad_files raises ValueError for empty radar."""
        with pytest.raises(ValueError, match="radar must be a 4-character string"):
            list_nexrad_files(
                radar="", start_time="2011-05-20 09:00", end_time="2011-05-20 11:00"
            )

    def test_list_nexrad_files_invalid_radar_wrong_length(self):
        """Test list_nexrad_files raises ValueError for wrong radar length."""
        with pytest.raises(ValueError, match="radar must be a 4-character string"):
            list_nexrad_files(
                radar="KVN",  # Too short
                start_time="2011-05-20 09:00",
                end_time="2011-05-20 11:00",
            )

    def test_list_nexrad_files_invalid_radar_none(self):
        """Test list_nexrad_files raises ValueError for None radar."""
        with pytest.raises(ValueError, match="radar must be a 4-character string"):
            list_nexrad_files(
                radar=None, start_time="2011-05-20 09:00", end_time="2011-05-20 11:00"
            )

    @patch("raw2zarr.utils.fsspec.filesystem")
    @patch("raw2zarr.utils._get_files_for_date")
    def test_list_nexrad_files_returns_sorted_results(
        self, mock_get_files, mock_filesystem
    ):
        """Test that list_nexrad_files returns sorted results."""
        mock_fs = Mock()
        mock_filesystem.return_value = mock_fs

        # Return files in non-sorted order
        mock_get_files.return_value = [
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_100147_V06",
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_110230_V06",
        ]

        result = list_nexrad_files(
            radar="KVNX", start_time="2011-05-20 09:00", end_time="2011-05-20 11:30"
        )

        # Should be sorted
        expected = [
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_093548_V06",
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_100147_V06",
            "s3://noaa-nexrad-level2/2011/05/20/KVNX/KVNX20110520_110230_V06",
        ]
        assert result == expected

    def test_list_nexrad_files_default_parameters(self):
        """Test list_nexrad_files with default parameters."""
        with (
            patch("raw2zarr.utils.fsspec.filesystem") as mock_filesystem,
            patch("raw2zarr.utils._get_files_for_date") as mock_get_files,
        ):

            mock_fs = Mock()
            mock_filesystem.return_value = mock_fs
            mock_get_files.return_value = []

            # Call with no parameters (should use defaults)
            result = list_nexrad_files()

            # Should use default parameters
            assert result == []
            mock_filesystem.assert_called_once_with("s3", anon=True)


# Pytest markers for organizing tests
class TestIntegration:
    """Integration tests that require network access (marked as slow)."""

    @pytest.mark.slow
    def test_load_vcp_samples_real_file(self):
        """Integration test with real VCP samples file."""
        # This test would require the actual file to exist
        # Skip if file doesn't exist in the test environment
        vcp_file = "data/vcp_samples.json"
        if not os.path.exists(vcp_file):
            pytest.skip("VCP samples file not found in test environment")

        result = load_vcp_samples(vcp_file)
        assert isinstance(result, dict)

    @pytest.mark.slow
    @pytest.mark.serial  # This test should not run in parallel
    def test_list_nexrad_files_real_s3_access(self):
        """Integration test with real S3 access (requires network)."""
        pytest.skip("Skipping real S3 access test to avoid network dependency")

        # This would test actual S3 access but we skip it for CI/CD
        # result = list_nexrad_files("KVNX", "2011-05-20 00:00", "2011-05-20 00:10")
        # assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__])
