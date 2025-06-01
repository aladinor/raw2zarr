from collections import defaultdict

import numpy as np
from packaging.version import parse as parse_version
from xarray import DataTree


def dtree_encoding(
    dtree: DataTree,
    append_dim: str,
) -> dict:
    """
    Encoding dictionary for time, append_dim, and all data variables within a radar DataTree.

    Parameters:
        dtree (DataTree): The xarray DataTree to process.
        append_dim (str): The dimension (e.g., 'vcp_time') to encode.

    Returns:
        dict: Dictionary suitable for use in xarray's `to_zarr` or similar export methods.
    """
    time_enc = {
        "units": "nanoseconds since 1950-01-01T00:00:00.00",
        "dtype": "int64",
        "_FillValue": -9999,
        "chunks": (1,),
    }
    encoding = defaultdict(dict)

    if not isinstance(dtree, DataTree):
        return {}

    for node in dtree.subtree:
        if node.is_empty:
            continue

        path = node.path
        ds = node.ds.copy(deep=True)

        if "time" in ds:
            encoding[path]["time"] = time_enc
        if append_dim in ds:
            encoding[path][append_dim] = time_enc

        for var_name, var in ds.data_vars.items():
            dims = var.dims
            dtype_kind = var.dtype.kind

            if dtype_kind in {"O", "U"}:
                encoding[path][var_name] = {
                    "dtype": "U50",
                    "chunks": (1,) * len(dims),
                    "_FillValue": "",
                }
            elif dims == (append_dim,):
                encoding[path][var_name] = {
                    "dtype": "float32" if dtype_kind == "f" else var.dtype,
                    "chunks": (1,),
                    "_FillValue": -9999,
                }
            elif set(dims) == {"azimuth", "range"}:
                az_chunksize = int((len(var["azimuth"])) // 2)
                range_chunksize = int(len(var.range) // 4)
                encoding[path][var_name] = {
                    "dtype": "float32" if dtype_kind == "f" else var.dtype,
                    "chunks": (
                        az_chunksize,
                        range_chunksize,
                    ),  # example, customize as needed
                    "_FillValue": -9999,
                }
            elif dims == ("range",):
                encoding[path][var_name] = {
                    "dtype": var.dtype,
                    "chunks": (len(ds["range"]),),
                    "_FillValue": -9999,
                }
            elif dims == ("azimuth",):
                encoding[path][var_name] = {
                    "dtype": var.dtype,
                    "chunks": (len(ds["azimuth"]),),
                    "_FillValue": -9999,
                }
            elif set(dims) == {append_dim, "azimuth", "range"}:
                az_chunksize = int((len(var["azimuth"])) // 2)
                range_chunksize = int(len(var.range) // 4)
                if dims != (append_dim, "azimuth", "range"):
                    var = var.transpose(append_dim, "azimuth", "range")
                    ds[var_name] = var
                encoding[path][var_name] = {
                    "dtype": "float32",
                    "chunks": (1, az_chunksize, range_chunksize),
                    "_FillValue": -999.0,
                }
            else:
                encoding[path][var_name] = {
                    "dtype": var.dtype,
                    "chunks": tuple(1 for _ in dims),
                    "_FillValue": -9999,
                }

    return dict(encoding)


def get_string_dtype():
    if parse_version(np.__version__) >= parse_version("2.0.0"):
        return np.dtypes.StringDType
    else:
        return np.dtype("U")
