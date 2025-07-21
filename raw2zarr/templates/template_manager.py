import json
from functools import lru_cache
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field

from ..transform.encoding import dtree_encoding
from .template_utils import (
    create_additional_groups,
    create_common_coords,
    create_root,
    remove_string_vars,
)

# from ..transform.alignment import fix_angle


class ScanCoordConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    dims: list[str]
    dtype: str
    attributes: dict = Field(default_factory=dict)


class ScanVariableConfig(ScanCoordConfig):
    fill_value: float = None


class ScanConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    # 🛠️ Added missing required fields from JSON structure
    dims: dict[str, int]
    coords: dict[str, ScanCoordConfig]
    variables: dict[str, ScanVariableConfig]
    metadata: dict[str, str | float]


class VcpConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    elevations: list[float]
    scan_types: list[str]
    dims: dict[str, list[int]]


class VcpTemplateManager:
    def __init__(
        self,
        scan_config_file: str = "scan_config.json",
        vcp_config_file: str = "vcp.json",
    ):
        config_dir = Path(__file__).resolve().parent.parent / "config"
        vcp_config_path = config_dir / vcp_config_file
        scan_config_path = config_dir / scan_config_file

        self.scan_config_path = scan_config_path
        self._full_config = None
        self.vcp_config_path = vcp_config_path
        self._vcp_configs = None

    @property
    def config(self):
        if self._full_config is None:
            with open(self.scan_config_path) as f:
                self._full_config = json.load(f)
        return self._full_config

    @property
    def vcp_config(self):
        if self._vcp_configs is None:
            with open(self.vcp_config_path) as f:
                self._vcp_configs = json.load(f)
        return self._vcp_configs

    @lru_cache(maxsize=32)
    def get_template(self, scan_type: str) -> ScanConfig:
        """Get validated config for specific scan type"""
        # 🛠️ Added error handling for missing scan types
        if scan_type not in self.config:
            raise ValueError(f"Scan type {scan_type} not found in config")
        return ScanConfig(**self.config[scan_type])

    @lru_cache(maxsize=32)
    def get_vcp_info(self, vcp: str) -> VcpConfig:
        """Get validated config for specific scan type"""
        # 🛠️ Added error handling for missing scan types
        if vcp not in self.vcp_config:
            raise ValueError(f"VCP {vcp} not found in config")
        return VcpConfig(**self.vcp_config[vcp])

    def create_scan_dataset(
        self,
        scan_type: str,
        sweep_idx: float,
        radar_info: dict,
        append_dim: str = "vcp_time",
        size_append_dim: int = 1,
        append_dim_time: list[pd.Timestamp] | None = None,
        dim_chunksize: dict = None,
    ) -> xr.Dataset:
        """Generic scan dataset creation"""
        cfg = self.get_template(scan_type)
        vcp = self.get_vcp_info(radar_info["vcp"])

        time_array = (
            np.array(append_dim_time, dtype="datetime64[ns]")
            if append_dim_time
            else np.array(range(size_append_dim), dtype="datetime64[ns]")
        )

        total_azimuth = vcp.dims["azimuth"][sweep_idx]
        total_bins = vcp.dims["range"][sweep_idx]
        elevation = vcp.elevations[sweep_idx]
        az_res = 360 / total_azimuth

        ds = xr.Dataset()
        coord_ds = create_common_coords(
            cfg=cfg,
            radar_info=radar_info,
            elevation=elevation,
            total_bins=total_bins,
            total_azimuth=total_azimuth,
            az_res=az_res,
            append_dim=append_dim,
            size_append_dim=size_append_dim,
            time_array=time_array,
        )
        ds.update(coord_ds)

        for var_name, var_cfg in cfg.variables.items():
            dims = (append_dim, "azimuth", "range")  # ensure consistent dimension order
            shape = (size_append_dim, total_azimuth, total_bins)

            ds[var_name] = xr.DataArray(
                da.full(shape, da.nan, dtype=var_cfg.dtype),
                dims=dims,
                attrs=var_cfg.attributes,
            )

        for var in cfg.metadata:
            ds[var] = xr.DataArray(
                da.from_array(
                    np.full(size_append_dim, cfg.metadata[var], dtype="U35"),
                    chunks=(1,),
                ),
                dims=(append_dim,),
            )

        ds["sweep_number"] = xr.DataArray(
            da.full((size_append_dim,), sweep_idx, dtype=float, chunks=(1,)),
            dims=(append_dim,),
        )

        ds["sweep_fixed_angle"] = xr.DataArray(
            da.full((size_append_dim,), elevation, dtype=float, chunks=(1,)),
            dims=(append_dim,),
        )

        ds = ds.set_coords(
            ["time", "longitude", "latitude", "altitude", "elevation", "crs_wkt"]
        )

        if dim_chunksize is None:
            az_chunksize = int(total_azimuth)
            range_chunksize = int(total_bins)
        else:
            az_chunksize = dim_chunksize.get("azimuth", int(total_azimuth))
            range_chunksize = dim_chunksize.get("range", int(total_bins))

        ds = ds.chunk(
            {"azimuth": az_chunksize, "range": range_chunksize, append_dim: 1}
        )

        ds = ds.xradar.georeference()
        ds["x"] = ds["x"].compute()
        ds["y"] = ds["y"].compute()
        ds["z"] = ds["z"].compute()
        return ds

    def create_empty_vcp_tree(
        self,
        radar_info: dict,
        append_dim: str,
        size_append_dim: int = 1,
        remove_strings: bool = True,  # remove after zarr v3 supports string dtypes
        append_dim_time: pd.Timestamp | None = None,
    ) -> xr.DataTree:

        vcp = radar_info["vcp"]
        vcp_info = self.get_vcp_info(vcp)

        root_ds = create_root(
            radar_info,
            append_dim=append_dim,
            size_append_dim=size_append_dim,
            append_dim_time=append_dim_time,
        )
        other_groups = create_additional_groups(
            vcp,
            append_dim=append_dim,
            size_append_dim=size_append_dim,
            radar_info=radar_info,
            append_dim_time=append_dim_time,
        )
        sweep_dict: dict = {}
        for sweep_idx in range(len(vcp_info.elevations)):
            scan_type = vcp_info.scan_types[sweep_idx]
            ds = self.create_scan_dataset(
                scan_type,
                sweep_idx,
                radar_info,
                append_dim=append_dim,
                size_append_dim=size_append_dim,
                append_dim_time=append_dim_time,
            )
            drop_vars = ["longitude", "latitude", "altitude", "crs_wkt"]
            sweep_dict[f"{vcp}/sweep_{sweep_idx}"] = ds.drop_vars(drop_vars)
        radar_dt = root_ds | other_groups | sweep_dict
        radar_dtree = xr.DataTree.from_dict(radar_dt)
        radar_dtree.encoding = dtree_encoding(
            radar_dtree,
            append_dim=append_dim,
        )
        # TODO: remove this when zarr v3 support string dtypes
        if remove_strings:
            clean_tree = remove_string_vars(radar_dtree)
            clean_tree.encoding = dtree_encoding(clean_tree, append_dim=append_dim)
            return clean_tree
        return radar_dtree
