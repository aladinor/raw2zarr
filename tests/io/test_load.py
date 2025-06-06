from unittest.mock import MagicMock, patch

import pytest

from raw2zarr.io.load import load_radar_data


class TestLoadRadarData:
    def test_with_iris(self):
        mock_loader = MagicMock(return_value="fake-datatree-iris")

        with patch.dict("raw2zarr.io.load.ENGINE_REGISTRY", {"iris": mock_loader}):
            result = load_radar_data("some/path/to.iris", engine="iris")

        mock_loader.assert_called_once_with("some/path/to.iris")
        assert result == "fake-datatree-iris"

    def test_with_nexradlevel2(self):
        mock_loader = MagicMock(return_value="fake-datatree-nexrad")

        with patch.dict(
            "raw2zarr.io.load.ENGINE_REGISTRY", {"nexradlevel2": mock_loader}
        ):
            result = load_radar_data("file.nexrad.gz", engine="nexradlevel2")

        mock_loader.assert_called_once_with("file.nexrad.gz")
        assert result == "fake-datatree-nexrad"

    def test_with_invalid_engine(self):
        with pytest.raises(ValueError, match="Unsupported engine 'unknown'"):
            load_radar_data("somefile", engine="unknown")
