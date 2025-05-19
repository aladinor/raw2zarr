import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataTree

from raw2zarr.transform.dimension import ensure_dimension, exp_dim


def create_dtree_without_dim() -> DataTree:
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


def create_dtree_with_dim(append_dim: str) -> DataTree:
    root_data = xr.Dataset(
        data_vars={
            "time_coverage_start": (append_dim, [pd.Timestamp("2023-01-01T00:00:00")])
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


def test_exp_dim_adds_dimension_and_coordinates():
    dtree = create_dtree_without_dim()
    result = exp_dim(dtree, append_dim="vcp_time")

    for node in result.subtree:
        assert "vcp_time" in node.ds.dims
        assert "vcp_time" in node.ds.coords
        assert isinstance(node.ds["vcp_time"].values[0], np.datetime64)


def test_ensure_dimension_adds_missing_dim():
    dtree = create_dtree_without_dim()
    result = ensure_dimension(dtree, append_dim="vcp_time")

    for node in result.subtree:
        assert "vcp_time" in node.ds.dims


def test_ensure_dimension_skips_if_dim_exists():
    dtree = create_dtree_with_dim("vcp_time")
    result = ensure_dimension(dtree, append_dim="vcp_time")

    for node in result.subtree:
        assert "vcp_time" in node.ds.dims
        # ensure no unintended reprocessing
        assert result == dtree
