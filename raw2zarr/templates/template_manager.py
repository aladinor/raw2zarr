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
        vcp_nexrad_config: str = "vcp_nexrad.json",
        scan_config_file: str = "scan_config.json",  # Backward compatibility
        vcp_config_file: str = "vcp.json",  # Backward compatibility
        scan_config_path: str = None,  # Backward compatibility
        vcp_config_path: str = None,  # Backward compatibility
    ):
        config_dir = Path(__file__).resolve().parent.parent / "config"

        # New unified config takes precedence
        self.vcp_nexrad_path = config_dir / vcp_nexrad_config

        # Backward compatibility: if unified config doesn't exist, use old two-file system
        if self.vcp_nexrad_path.exists():
            self._use_unified_config = True
            self._unified_config = None
        else:
            self._use_unified_config = False
            # Support both parameter styles for backward compatibility
            if scan_config_path is not None:
                scan_config_path = Path(scan_config_path)
            else:
                scan_config_path = config_dir / scan_config_file

            if vcp_config_path is not None:
                vcp_config_path = Path(vcp_config_path)
            else:
                vcp_config_path = config_dir / vcp_config_file

            self.scan_config_path = scan_config_path
            self._full_config = None
            self.vcp_config_path = vcp_config_path
            self._vcp_configs = None

    @property
    def unified_config(self):
        """Get unified VCP configuration (new system)"""
        if self._use_unified_config:
            if self._unified_config is None:
                with open(self.vcp_nexrad_path) as f:
                    self._unified_config = json.load(f)
            return self._unified_config
        return None

    @property
    def config(self):
        """Get scan config (backward compatibility)"""
        if not self._use_unified_config:
            if self._full_config is None:
                with open(self.scan_config_path) as f:
                    self._full_config = json.load(f)
            return self._full_config
        return None

    @property
    def vcp_config(self):
        """Get VCP config (backward compatibility)"""
        if not self._use_unified_config:
            if self._vcp_configs is None:
                with open(self.vcp_config_path) as f:
                    self._vcp_configs = json.load(f)
            return self._vcp_configs
        return None

    def get_sweep_config(self, vcp: str, sweep_idx: int) -> dict:
        """Get sweep configuration from unified config (new system)"""
        if self._use_unified_config:
            unified = self.unified_config
            if vcp not in unified:
                raise ValueError(f"VCP {vcp} not found in unified config")

            sweep_key = f"sweep_{sweep_idx}"
            if sweep_key not in unified[vcp]:
                raise ValueError(f"Sweep {sweep_idx} not found in {vcp}")

            return unified[vcp][sweep_key]
        else:
            raise ValueError("get_sweep_config only available with unified config")

    @lru_cache(maxsize=32)
    def get_template(self, scan_type: str) -> ScanConfig:
        """Get validated config for specific scan type (backward compatibility)"""
        if self._use_unified_config:
            raise ValueError(
                "get_template not available with unified config. Use get_sweep_config instead."
            )

        # 🛠️ Added error handling for missing scan types
        if scan_type not in self.config:
            raise ValueError(f"Scan type {scan_type} not found in config")
        return ScanConfig(**self.config[scan_type])

    @lru_cache(maxsize=32)
    def get_vcp_info(self, vcp: str) -> VcpConfig:
        """Get validated config for specific VCP"""
        if self._use_unified_config:
            # Extract basic VCP info from unified config
            unified = self.unified_config
            if vcp not in unified:
                raise ValueError(f"VCP {vcp} not found in unified config")

            vcp_data = unified[vcp]

            # For backward compatibility, create scan_types list based on sweep patterns
            scan_types = []
            elevations = vcp_data.get("elevations", [])
            dims = vcp_data.get("dims", {"azimuth": [], "range": []})

            # Create dummy scan types for unified config (not used in new system)
            for i in range(len(elevations)):
                scan_types.append(f"unified_sweep_{i}")

            return VcpConfig(elevations=elevations, scan_types=scan_types, dims=dims)
        else:
            # 🛠️ Added error handling for missing scan types
            if vcp not in self.vcp_config:
                raise ValueError(f"VCP {vcp} not found in config")
            return VcpConfig(**self.vcp_config[vcp])

    def create_scan_dataset(
        self,
        scan_type: str,
        sweep_idx: int,
        radar_info: dict,
        append_dim: str = "vcp_time",
        size_append_dim: int = 1,
        append_dim_time: list[pd.Timestamp] | None = None,
        dim_chunksize: dict = None,
    ) -> xr.Dataset:
        """Generic scan dataset creation"""

        if self._use_unified_config:
            # New unified config system - get variables directly from sweep config
            vcp_name = radar_info["vcp"]
            sweep_config = self.get_sweep_config(vcp_name, sweep_idx)
            vcp_info = self.get_vcp_info(vcp_name)

            time_array = (
                np.array(append_dim_time, dtype="datetime64[ns]")
                if append_dim_time
                else np.array(range(size_append_dim), dtype="datetime64[ns]")
            )

            total_azimuth = vcp_info.dims["azimuth"][sweep_idx]
            total_bins = vcp_info.dims["range"][sweep_idx]
            elevation = vcp_info.elevations[sweep_idx]
            az_res = 360 / total_azimuth

            # Create a dummy ScanConfig for coordinate creation
            dummy_cfg = type(
                "DummyConfig",
                (),
                {
                    "coords": {
                        "range": type(
                            "RangeConfig",
                            (),
                            {
                                "dims": ["range"],
                                "dtype": "float32",
                                "attributes": {
                                    "standard_name": "projection_range_coordinate",
                                    "long_name": "range_to_measurement_volume",
                                    "units": "meters",
                                    "axis": "radial_range_coordinate",
                                    "meters_between_gates": 250.0,
                                    "spacing_is_constant": "true",
                                    "meters_to_center_of_first_gate": 2125.0,
                                },
                            },
                        )(),
                        "azimuth": type(
                            "AzimuthConfig",
                            (),
                            {
                                "dims": ["azimuth"],
                                "dtype": "float64",
                                "attributes": {
                                    "standard_name": "ray_azimuth_angle",
                                    "long_name": "azimuth_angle_from_true_north",
                                    "units": "degrees",
                                    "axis": "radial_azimuth_coordinate",
                                },
                            },
                        )(),
                        "elevation": type(
                            "ElevationConfig",
                            (),
                            {
                                "dims": ["azimuth"],
                                "dtype": "float64",
                                "attributes": {
                                    "standard_name": "ray_elevation_angle",
                                    "long_name": "elevation_angle_from_horizontal_plane",
                                    "units": "degrees",
                                    "axis": "radial_elevation_coordinate",
                                },
                            },
                        )(),
                    },
                    "metadata": {
                        "sweep_mode": sweep_config.get(
                            "scan_mode", "azimuth_surveillance"
                        )
                    },
                },
            )()

            # Use existing coordinate creation function
            coord_ds = create_common_coords(
                cfg=dummy_cfg,
                radar_info=radar_info,
                elevation=elevation,
                total_bins=total_bins,
                total_azimuth=total_azimuth,
                az_res=az_res,
                append_dim=append_dim,
                size_append_dim=size_append_dim,
                time_array=time_array,
            )

            # Create data variables directly from sweep config
            data_vars = {}
            for var_name, var_config in sweep_config["variables"].items():
                dims = (append_dim, "azimuth", "range")
                shape = (size_append_dim, total_azimuth, total_bins)

                data_vars[var_name] = xr.DataArray(
                    da.full(shape, da.nan, dtype=var_config["dtype"]),
                    dims=dims,
                    attrs=var_config["attributes"],
                )

            # Create dataset
            ds = xr.Dataset(data_vars, coords=coord_ds)

            # Add metadata
            ds["sweep_mode"] = xr.DataArray(
                da.from_array(
                    np.full(
                        size_append_dim,
                        sweep_config.get("scan_mode", "azimuth_surveillance"),
                        dtype="U35",
                    ),
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

        else:
            # Legacy two-file system
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

            # Create coords first with VCP-specific dimensions
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

            # Create data variables with VCP-specific dimensions
            data_vars = {}
            for var_name, var_cfg in cfg.variables.items():
                dims = (
                    append_dim,
                    "azimuth",
                    "range",
                )  # ensure consistent dimension order
                shape = (size_append_dim, total_azimuth, total_bins)

                data_vars[var_name] = xr.DataArray(
                    da.full(shape, da.nan, dtype=var_cfg.dtype),
                    dims=dims,
                    attrs=var_cfg.attributes,
                )

            # Create dataset with all components at once to avoid alignment issues
            ds = xr.Dataset(data_vars, coords=coord_ds)

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

        # Common chunking and georeferencing
        if dim_chunksize is None:
            az_chunksize = int(total_azimuth)
            range_chunksize = int(total_bins)
        else:
            az_chunksize = dim_chunksize.get("azimuth", int(total_azimuth))
            range_chunksize = dim_chunksize.get("range", int(total_bins))

        ds = ds.chunk(
            {"azimuth": az_chunksize, "range": range_chunksize, append_dim: 1}
        )

        with xr.set_options(keep_attrs=True):
            ds = ds.xradar.georeference()
            ds["x"] = ds["x"].compute()
            ds["y"] = ds["y"].compute()
            ds["z"] = ds["z"].compute()
            ds["longitude"] = ds["longitude"].compute()
            ds["latitude"] = ds["latitude"].compute()
            ds["altitude"] = ds["altitude"].compute()
            ds["crs_wkt"] = ds["crs_wkt"].compute()
        return ds

    def create_empty_vcp_tree(
        self,
        radar_info: dict,
        append_dim: str,
        remove_strings: bool = True,  # remove after zarr v3 supports string dtypes
        append_dim_time: pd.Timestamp | None = None,
    ) -> xr.DataTree:

        vcp = radar_info["vcp"]
        vcp_info = self.get_vcp_info(vcp)
        size_append_dim = len(append_dim_time)
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
            if self._use_unified_config:
                scan_type = (
                    f"unified_sweep_{sweep_idx}"  # Placeholder for unified config
                )
            else:
                scan_type = vcp_info.scan_types[sweep_idx]
            ds = self.create_scan_dataset(
                scan_type,
                sweep_idx,
                radar_info,
                append_dim=append_dim,
                size_append_dim=size_append_dim,
                append_dim_time=append_dim_time,
            )
            # Keep scalar coordinates in sweep datasets for proper region writing
            sweep_dict[f"{vcp}/sweep_{sweep_idx}"] = ds
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

    def create_multi_vcp_tree(
        self,
        radar_info: dict,
        vcp_time_mapping: dict,
        append_dim: str,
        remove_strings: bool = True,
        dim_chunksize: dict = None,
    ) -> xr.DataTree:
        """
        Create a DataTree template that supports multiple VCP patterns.

        Each VCP gets its own time block within the append dimension.

        Parameters:
            radar_info: Basic radar metadata (location, instrument, etc.)
            vcp_time_mapping: VCP-time mapping from _create_vcp_time_mapping()
            append_dim: Time dimension name (e.g., 'vcp_time')
            remove_strings: Whether to remove string variables
            dim_chunksize: Optional custom chunk sizes

        Returns:
            xr.DataTree: Multi-VCP template with hierarchical structure
        """
        # Calculate total time dimension size across all VCPs
        total_time_size = sum(info["file_count"] for info in vcp_time_mapping.values())

        # Create consolidated timestamp array for all VCPs
        all_timestamps = []
        for vcp_name in sorted(vcp_time_mapping.keys()):
            all_timestamps.extend(vcp_time_mapping[vcp_name]["timestamps"])

        # Create consolidated timestamp array for all VCPs
        all_timestamps = []
        for vcp_name in sorted(vcp_time_mapping.keys()):
            all_timestamps.extend(vcp_time_mapping[vcp_name]["timestamps"])

        # Create VCP-level groups using individual VCP time dimensions (working approach)
        vcp_groups = {}
        for vcp_name in vcp_time_mapping.keys():
            vcp_info = vcp_time_mapping[vcp_name]
            vcp_radar_info = radar_info.copy()
            vcp_radar_info["vcp"] = vcp_name

            # Use the original create_root approach for each VCP
            from .template_utils import create_additional_groups, create_root

            vcp_root_dict = create_root(
                vcp_radar_info,
                append_dim=append_dim,
                size_append_dim=total_time_size,  # Each VCP gets full time dimension for region writing
                append_dim_time=all_timestamps,  # Use all timestamps for consistent structure
            )

            # Extract the VCP dataset from the dictionary (remove the VCP key wrapper)
            vcp_groups[vcp_name] = list(vcp_root_dict.values())[0]

            # Add additional groups for this VCP
            additional_groups = create_additional_groups(
                vcp_name,
                append_dim=append_dim,
                size_append_dim=total_time_size,  # Each VCP gets full time dimension for region writing
                radar_info=vcp_radar_info,
                append_dim_time=all_timestamps,
            )

            # Add additional groups to the structure (they'll be added later in radar_dt)
            for group_path, group_ds in additional_groups.items():
                # Extract just the group name (remove VCP prefix)
                group_name = group_path.split("/")[-1]
                vcp_groups[f"{vcp_name}/{group_name}"] = group_ds

        # Create sweep datasets for each VCP
        sweep_dict = {}

        for vcp_name, vcp_info in vcp_time_mapping.items():
            vcp_config = self.get_vcp_info(vcp_name)

            # Update radar_info for this specific VCP
            vcp_radar_info = radar_info.copy()
            vcp_radar_info["vcp"] = vcp_name

            print(
                f"  📡 Creating template for {vcp_name}: {len(vcp_config.elevations)} sweeps"
            )

            # Create sweeps for this VCP using the FULL time dimension
            for sweep_idx in range(len(vcp_config.elevations)):
                if self._use_unified_config:
                    scan_type = (
                        f"unified_sweep_{sweep_idx}"  # Placeholder for unified config
                    )
                else:
                    scan_type = vcp_config.scan_types[sweep_idx]

                # Clear dask caches to avoid task key conflicts between VCPs/sweeps
                import gc

                try:
                    import dask

                    dask.base.clear_cache()
                except (ImportError, AttributeError):
                    pass
                gc.collect()

                # Create dataset for this sweep with TOTAL time dimension
                # The region writing will handle placing data in correct time slices
                ds = self.create_scan_dataset(
                    scan_type,
                    sweep_idx,
                    vcp_radar_info,
                    append_dim=append_dim,
                    size_append_dim=total_time_size,  # Use TOTAL time size across all VCPs
                    append_dim_time=all_timestamps,  # Use ALL timestamps from all VCPs
                    dim_chunksize=dim_chunksize,
                )

                # Keep scalar coordinates in sweep datasets for proper region writing
                sweep_dict[f"{vcp_name}/sweep_{sweep_idx}"] = ds

                print(
                    f"    🔹 Sweep {sweep_idx}: {vcp_config.elevations[sweep_idx]:.1f}° | {scan_type} | "
                    f"Az:{vcp_config.dims['azimuth'][sweep_idx]} R:{vcp_config.dims['range'][sweep_idx]}"
                )

        # Combine all components into proper multi-VCP structure (no root dataset)
        radar_dt = {
            **{
                f"/{vcp_name}": vcp_ds
                for vcp_name, vcp_ds in vcp_groups.items()
                if "/" not in vcp_name
            },  # VCP groups
            **{f"/{path}": ds for path, ds in sweep_dict.items()},  # Sweep datasets
        }
        radar_dtree = xr.DataTree.from_dict(radar_dt)

        # Apply encoding
        radar_dtree.encoding = dtree_encoding(
            radar_dtree,
            append_dim=append_dim,
            dim_chunksize=dim_chunksize,
        )

        # Remove string variables if requested
        if remove_strings:
            clean_tree = remove_string_vars(radar_dtree)
            clean_tree.encoding = dtree_encoding(
                clean_tree,
                append_dim=append_dim,
                dim_chunksize=dim_chunksize,
            )
            return clean_tree

        return radar_dtree
