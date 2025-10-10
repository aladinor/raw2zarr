"""Tests for template utility functions."""

import numpy as np
import pandas as pd
import pytest

from raw2zarr.templates.template_utils import _convert_timestamps_to_datetime64


class TestConvertTimestampsToDatetime64:
    """Test timezone-aware timestamp conversion to numpy datetime64."""

    def test_utc_timezone_conversion(self):
        """Test conversion of UTC timestamps removes timezone info."""
        timestamps = [
            pd.Timestamp("2023-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2023-01-01 12:05:00", tz="UTC"),
        ]
        result = _convert_timestamps_to_datetime64(timestamps)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.dtype("datetime64[ns]")
        assert len(result) == 2
        # Verify values preserved (no timezone conversion needed for UTC)
        assert result[0] == np.datetime64("2023-01-01T12:00:00")
        assert result[1] == np.datetime64("2023-01-01T12:05:00")

    def test_non_utc_timezone_converts_to_utc(self):
        """Test non-UTC timezones are converted to UTC."""
        # US/Eastern is UTC-5 (EST) or UTC-4 (EDT)
        # Using a date in January when EST is in effect (UTC-5)
        timestamps = [pd.Timestamp("2023-01-01 12:00:00", tz="US/Eastern")]
        result = _convert_timestamps_to_datetime64(timestamps)

        # 12:00 EST (UTC-5) = 17:00 UTC
        assert result[0] == np.datetime64("2023-01-01T17:00:00")

    def test_naive_timestamps(self):
        """Test timezone-naive timestamps work without conversion."""
        timestamps = [
            pd.Timestamp("2023-01-01 12:00:00"),
            pd.Timestamp("2023-01-01 13:00:00"),
        ]
        result = _convert_timestamps_to_datetime64(timestamps)

        assert result[0] == np.datetime64("2023-01-01T12:00:00")
        assert result[1] == np.datetime64("2023-01-01T13:00:00")

    def test_none_fallback_to_range(self):
        """Test None timestamps fall back to range array."""
        result = _convert_timestamps_to_datetime64(None, size=5)

        assert len(result) == 5
        expected = np.array(range(5), dtype="datetime64[ns]")
        np.testing.assert_array_equal(result, expected)

    def test_empty_list_falls_back_to_range(self):
        """Test empty list falls back to range array."""
        result = _convert_timestamps_to_datetime64([])

        # Empty list evaluates to False, so falls back to range(1)
        assert len(result) == 1
        assert result.dtype == np.dtype("datetime64[ns]")

    def test_no_userwarning_for_timezone_aware(self):
        """Verify no UserWarning about timezones when converting."""
        import warnings

        timestamps = [pd.Timestamp("2023-01-01 12:00:00", tz="UTC")]

        # Capture all warnings
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            _convert_timestamps_to_datetime64(timestamps)

        # Filter for timezone-related warnings
        tz_warnings = [w for w in warning_list if "timezone" in str(w.message).lower()]
        assert len(tz_warnings) == 0, "Should not produce timezone warnings"

    def test_mixed_timezone_aware_and_naive_raises_error(self):
        """Test that mixing timezone-aware and naive timestamps raises error."""
        timestamps = [
            pd.Timestamp("2023-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2023-01-01 13:00:00"),  # Naive
        ]

        # pandas DatetimeIndex should raise when mixing aware and naive
        with pytest.raises((TypeError, ValueError)):
            _convert_timestamps_to_datetime64(timestamps)

    def test_single_timezone_converts_to_utc(self):
        """Test single non-UTC timezone converts to UTC correctly."""
        timestamps = [
            pd.Timestamp("2023-01-01 12:00:00", tz="US/Eastern"),  # UTC-5 = 17:00 UTC
            pd.Timestamp("2023-01-01 13:00:00", tz="US/Eastern"),  # UTC-5 = 18:00 UTC
        ]

        result = _convert_timestamps_to_datetime64(timestamps)

        # Verify conversion to UTC
        assert result[0] == np.datetime64("2023-01-01T17:00:00")
        assert result[1] == np.datetime64("2023-01-01T18:00:00")
