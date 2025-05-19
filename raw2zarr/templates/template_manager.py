import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field


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


class ScanTemplateManager:
    def __init__(
        self,
        scan_config_path: Path = Path("../config/scan_config.json"),
        vcp_config_path: Path = Path("../config/vcp.json"),
    ):
        self.config_path = scan_config_path
        self._full_config = None
        self.vcp_config_path = vcp_config_path
        self._vcp_configs = None

    @property
    def config(self):
        if self._full_config is None:
            with open(self.config_path) as f:
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
        self, scan_type: str, sweep_idx: float, radar_info: dict
    ) -> xr.Dataset:
        """Generic scan dataset creation"""
        cfg = self.get_template(scan_type)
        vcp = self.get_vcp_info(radar_info["vcp"])
        elevation = vcp.elevations[sweep_idx]
        ds = xr.Dataset()
        total_azimuth = vcp.dims["azimuth"][sweep_idx]
        total_bins = vcp.dims["range"][sweep_idx]
        az_res = 360 / total_azimuth
        ds["azimuth"] = xr.DataArray(
            np.arange(
                az_res / 2,
                cfg.dims["azimuth"],
                az_res,
                dtype=cfg.coords["azimuth"].dtype,
            ),
            dims="azimuth",
            attrs=cfg.coords["azimuth"].attributes,
        )
        # Add range coordinate

        ds["range"] = xr.DataArray(
            np.arange(
                cfg.coords["range"].attributes["meters_to_center_of_first_gate"],
                cfg.coords["range"].attributes["meters_to_center_of_first_gate"]
                + cfg.coords["range"].attributes["meters_between_gates"] * total_bins,
                cfg.coords["range"].attributes["meters_between_gates"],
                dtype=cfg.coords["range"].dtype,
            ),
            dims="range",
            attrs=cfg.coords["range"].attributes,
        )

        # Add radar location
        ds["longitude"] = xr.DataArray(
            radar_info["lon"], attrs={"standard_name": "longitude"}
        )

        ds["latitude"] = xr.DataArray(
            radar_info["lat"], attrs={"standard_name": "latitude"}
        )

        ds["altitude"] = xr.DataArray(
            radar_info["alt"], attrs={"standard_name": "altitude"}
        )

        for var_name, var_cfg in cfg.variables.items():
            ds[var_name] = xr.DataArray(
                np.full([total_azimuth, total_bins], np.nan, dtype=var_cfg.dtype),
                dims=var_cfg.dims,
                attrs=var_cfg.attributes,
            )

        # Add time coordinates
        ds["time"] = xr.DataArray(
            np.full(
                total_azimuth,
                radar_info["reference_time"],
                dtype="datetime64[ns]",
            ),
            dims="azimuth",
            attrs={"standard_name": "time"},
        )

        # Add elevation array
        ds["elevation"] = xr.DataArray(
            np.full(cfg.dims["azimuth"], elevation, dtype=np.float64),
            dims="azimuth",
            attrs=cfg.coords["elevation"].attributes,
        )
        # TO DO: add follow_mode, prt_mode, sweep_mode, sweep_fixed_angle

        ds = ds.set_coords(["time", "longitude", "latitude", "altitude", "elevation"])
        return ds.xradar.georeference()
