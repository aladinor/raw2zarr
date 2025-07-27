import pytest

import raw2zarr.transform.tranform_utils as transform_utils
from raw2zarr.transform.tranform_utils import get_vcp_values


class TestGetVCPValues:
    def test_get_vcp_values_valid(self):
        vcp = get_vcp_values("VCP-212")
        assert isinstance(vcp, list)
        assert all(isinstance(e, (float, int)) for e in vcp)
        assert len(vcp) > 0

    def test_get_vcp_values_missing_key(self):
        with pytest.raises(KeyError, match="VCP 'FAKE-VCP' not found"):
            get_vcp_values("FAKE-VCP")

    def test_get_vcp_values_invalid_structure(self, monkeypatch):
        # Mock VcpTemplateManager to return invalid config data
        class MockVcpInfo:
            def __init__(self):
                self.elevations = "not_a_list"  # Invalid - should be a list

        class MockTemplateManager:
            def get_vcp_info(self, vcp_name):
                return MockVcpInfo()

        monkeypatch.setattr(transform_utils, "VcpTemplateManager", MockTemplateManager)

        with pytest.raises(ValueError, match="Invalid 'elevations' list"):
            transform_utils.get_vcp_values("VCP-212")
