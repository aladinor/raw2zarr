import pytest

import raw2zarr.transform.utils as transform_utils
from raw2zarr.transform.alignment import get_vcp_values


def test_get_vcp_values_valid():
    vcp = get_vcp_values("VCP-212")
    assert isinstance(vcp, list)
    assert all(isinstance(e, (float, int)) for e in vcp)
    assert len(vcp) > 0


def test_get_vcp_values_missing_key():
    with pytest.raises(KeyError, match="VCP 'FAKE-VCP' not found"):
        get_vcp_values("FAKE-VCP")


def test_get_vcp_values_invalid_structure(monkeypatch):
    def mock_bad_config(_):
        return {"VCP-212": {"elevations": "not_a_list"}}

    monkeypatch.setattr(transform_utils, "load_json_config", mock_bad_config)

    with pytest.raises(ValueError, match="Invalid 'elevations' list"):
        transform_utils.get_vcp_values("VCP-212")
