from collections import defaultdict

import numpy as np
from xarray import DataTree


def dtree_encoding(dtree: DataTree, append_dim: str) -> dict:
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
    }
    var_enc = {"_FillValue": -9999}
    encoding = defaultdict(dict)

    if not isinstance(dtree, DataTree):
        return {}

    for node in dtree.subtree:
        if node.is_empty:
            continue

        path = node.path
        ds = node.ds

        if "time" in ds:
            encoding[path]["time"] = time_enc
        if append_dim in ds:
            encoding[path][append_dim] = time_enc

        for var_name, var in ds.data_vars.items():
            if var.dtype.kind in {"O", "U"}:
                encoding[path][var_name] = {"dtype": np.dtypes.StringDType}
            else:
                encoding[path][var_name] = var_enc

    return dict(encoding)
