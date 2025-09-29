import dask.array as da
import numpy as np
import xarray as xr
from pandas import Timestamp


def create_common_coords(
    cfg,
    radar_info: dict,
    elevation: float,
    total_bins: int,
    total_azimuth: int,
    append_dim: str,
    az_res: float = 0.5,
    size_append_dim: int = 1,
    time_array=list[Timestamp],
):
    """Create coordinate and metadata variables shared across scan datasets."""
    ds = xr.Dataset()

    ds["azimuth"] = xr.DataArray(
        da.linspace(
            az_res / 2,
            360 - az_res / 2,
            total_azimuth,
            dtype=cfg.coords["azimuth"].dtype,
        ),
        dims="azimuth",
        attrs=cfg.coords["azimuth"].attributes,
    ).values

    ds["range"] = xr.DataArray(
        da.arange(
            cfg.coords["range"].attributes["meters_to_center_of_first_gate"],
            cfg.coords["range"].attributes["meters_to_center_of_first_gate"]
            + cfg.coords["range"].attributes["meters_between_gates"] * total_bins,
            cfg.coords["range"].attributes["meters_between_gates"],
            dtype=cfg.coords["range"].dtype,
        ),
        dims="range",
        attrs=cfg.coords["range"].attributes,
    )

    ds["altitude"] = xr.DataArray(
        da.array(radar_info["alt"], dtype=float),
        attrs={"long_name": "altitude", "units": "meters", "standard_name": "altitude"},
    )

    ds["elevation"] = xr.DataArray(
        da.full_like(ds["azimuth"], elevation, dtype=float),
        dims="azimuth",
        attrs=cfg.coords["elevation"].attributes,
    )

    ds[append_dim] = xr.DataArray(
        da.array(time_array, dtype="datetime64[ns]"),
        dims=append_dim,
        coords={append_dim: range(size_append_dim)},
        attrs={
            "description": "Volume Coverage Pattern time since start of volume scan"
        },
    ).chunk({append_dim: 1})

    # Only add crs_wkt if it exists in radar_info
    if "crs_wkt" in radar_info:
        ds["crs_wkt"] = xr.DataArray(
            da.array(0, dtype=int), attrs=radar_info["crs_wkt"]
        )

    ds["longitude"] = xr.DataArray(
        da.array(radar_info["lon"], dtype=float),
        attrs={
            "standard_name": "longitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "positive": "up",
        },
    )

    ds["latitude"] = xr.DataArray(
        da.array(radar_info["lat"], dtype=float),
        attrs={
            "long_name": "latitude",
            "units": "degrees_north",
            "positive": "up",
            "standard_name": "latitude",
        },
    )

    ds["time"] = xr.DataArray(
        da.full_like(
            ds["azimuth"], radar_info["reference_time"], dtype="datetime64[ns]"
        ),
        dims="azimuth",
        attrs={"standard_name": "time"},
    )
    az_chunksize = int(total_azimuth // 2)
    range_chunksize = int(total_bins // 4)

    return ds.chunk({"azimuth": az_chunksize, "range": range_chunksize, "vcp_time": 1})


def create_root(
    radar_info: dict,
    append_dim: str,
    size_append_dim: int = 1,
    append_dim_time: list[Timestamp] | None = None,
) -> dict[str : xr.Dataset]:
    time_array = (
        np.array(append_dim_time, dtype="datetime64[ns]")
        if append_dim_time
        else np.array(range(size_append_dim), dtype="datetime64[ns]")
    )
    vpc_coord = xr.DataArray(
        da.from_array(time_array, chunks=1),
        dims=append_dim,
        attrs={
            "description": "Volume Coverage Pattern time since start of volume scan"
        },
    )

    root_ds = xr.Dataset(
        data_vars={
            "vcp_time": vpc_coord,
            "volume_number": (
                (append_dim,),
                da.full(
                    (size_append_dim,),
                    int(radar_info["volume_number"]),
                    dtype=int,
                    chunks=(1,),
                ),
            ),
            "platform_type": (
                (append_dim,),
                da.from_array(
                    np.array(
                        [radar_info["platform_type"]] * size_append_dim,
                        dtype="U25",
                    ),
                    chunks=(1,),
                ),
            ),
            "instrument_type": (
                (append_dim,),
                da.from_array(
                    np.array(
                        [radar_info["instrument_type"]] * size_append_dim,
                        dtype="U25",
                    ),
                    chunks=(1,),
                ),
            ),
            "time_coverage_start": (
                (append_dim,),
                da.from_array(
                    np.array(
                        [radar_info["time_coverage_start"]] * size_append_dim,
                        dtype="U25",
                    ),
                    chunks=(1,),
                ),
            ),
            "time_coverage_end": (
                (append_dim,),
                da.from_array(
                    np.array(
                        [radar_info["time_coverage_end"]] * size_append_dim,
                        dtype="U25",
                    ),
                    chunks=(1,),
                ),
            ),
            "longitude": (
                (append_dim,),
                da.full(
                    (size_append_dim,), radar_info["lon"], dtype=float, chunks=(1,)
                ),
            ),
            "latitude": (
                (append_dim,),
                da.full(
                    (size_append_dim,), radar_info["lat"], dtype=float, chunks=(1,)
                ),
            ),
            "altitude": (
                (append_dim,),
                da.full(
                    (size_append_dim,), radar_info["alt"], dtype=float, chunks=(1,)
                ),
            ),
        },
        coords={},
        attrs={
            "Conventions": "",
            "instrument_name": radar_info.get("instrument_name", "Unknown"),
            "version": "",
            "title": "",
            "institution": "",
            "references": "",
            "source": "",
            "history": "",
            "comment": "",
            "scan_name": radar_info["vcp"],
        },
    )
    return {radar_info["vcp"]: root_ds.set_coords(append_dim)}


def create_additional_groups(
    vcp: str,
    append_dim: str,
    radar_info: dict,
    size_append_dim: int = 1,
    append_dim_time: Timestamp | None = None,
) -> dict[str : xr.Dataset]:
    time_array = (
        np.array(append_dim_time, dtype="datetime64[ns]")
        if append_dim_time
        else np.array(range(size_append_dim), dtype="datetime64[ns]")
    )

    group_names = ["georeferencing_correction", "radar_parameters"]
    additional_groups = {}
    for group_name in group_names:
        additional_groups[f"{vcp}/{group_name}"] = xr.Dataset(
            coords={
                "altitude": (da.array(radar_info["alt"], dtype=int)),
                "latitude": (da.array(radar_info["lat"], dtype=float)),
                "longitude": (da.array(radar_info["lat"], dtype=float)),
                append_dim: (
                    append_dim,
                    da.from_array(
                        time_array,
                        chunks=1,
                    ),
                ),
            }
        )
    additional_groups[f"{vcp}/radar_calibration"] = xr.Dataset()

    return additional_groups


def remove_string_vars(dtree: xr.DataTree) -> xr.DataTree:
    def remove_str(ds: xr.Dataset) -> xr.Dataset:
        return ds.drop_vars(
            [var for var in ds.data_vars if ds[var].dtype.kind in {"O", "U"}]
        )

    return dtree.map_over_datasets(remove_str)


def align_azimuth_dim(dtree: xr.DataTree) -> xr.DataTree:
    from ..transform.alignment import reindex_angle

    return dtree.map_over_datasets(reindex_angle)
