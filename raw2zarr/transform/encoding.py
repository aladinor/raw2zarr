from collections import defaultdict
from collections.abc import Generator

import numpy as np
from packaging.version import parse as parse_version
from xarray import DataTree


def _iter_dtree_nodes(dtree: DataTree) -> Generator[DataTree, None, None]:
    """
    Iterate over all nodes in a DataTree, similar to xarray's _iter_zarr_groups.

    This ensures we capture all groups in multi-VCP structures where
    dtree.subtree might miss some nodes.
    """
    yield dtree
    for child in dtree.children.values():
        yield from _iter_dtree_nodes(child)


def dtree_encoding(
    dtree: DataTree,
    append_dim: str,
    dim_chunksize: dict = None,
) -> dict:
    """
    Encoding dictionary for time, append_dim, and all data variables within a radar DataTree.

    Parameters:
        dtree (DataTree): The xarray DataTree to process.
        append_dim (str): The dimension (e.g., 'vcp_time') to encode.
        dim_chunksize (dict, optional): Custom chunk sizes for dimensions. If None, uses coordinate lengths.

    Returns:
        dict: Dictionary suitable for use in xarray's `to_zarr` or similar export methods.
    """
    # Provide default chunk sizes to avoid None checks
    if dim_chunksize is None:
        dim_chunksize = {}

    append_dim_chunzise = dim_chunksize.get(append_dim, 1_000_000)
    time_enc = {
        "units": "nanoseconds since 1950-01-01T00:00:00.00",
        "dtype": "int64",
        "_FillValue": -9999,
        "chunks": (append_dim_chunzise,),
    }
    encoding = defaultdict(dict)

    if not isinstance(dtree, DataTree):
        return {}

    # Use proper iteration over all groups, similar to xarray's internal _iter_zarr_groups
    for node in _iter_dtree_nodes(dtree):
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

            def get_chunk_sizes():
                az_chunksize = dim_chunksize.get("azimuth", int(len(var["azimuth"])))
                range_chunksize = dim_chunksize.get("range", int(len(var["range"])))
                return az_chunksize, range_chunksize

            if dtype_kind in {"O", "U"}:
                # Use the actual dtype from the variable, not hardcoded U50
                encoding[path][var_name] = {
                    "dtype": var.dtype,
                    "chunks": (1,) * len(dims),
                    # TODO fix this after zarrv3 string enconding dtype is accepted
                    # "_FillValue": "None",
                }
            elif dims == (append_dim,):
                encoding[path][var_name] = {
                    "dtype": "float32" if dtype_kind == "f" else var.dtype,
                    "chunks": (1,),
                    "_FillValue": -9999,
                }
            elif set(dims) == {"azimuth", "range"}:
                az_chunksize, range_chunksize = get_chunk_sizes()
                encoding[path][var_name] = {
                    "dtype": "float32" if dtype_kind == "f" else var.dtype,
                    "chunks": (az_chunksize, range_chunksize),
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
                az_chunksize, range_chunksize = get_chunk_sizes()
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
