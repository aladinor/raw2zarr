import pandas as pd
from xarray import DataTree


def exp_dim(dtree: DataTree, append_dim: str) -> DataTree:
    """
    Add a new dimension to all datasets in a DataTree and initialize it with a specific value.

    Parameters:
        dtree (DataTree): The DataTree containing radar datasets.
        append_dim (str): The name of the dimension to add.

    Returns:
        DataTree: A DataTree with the specified dimension added to all datasets.
    """
    start_time = pd.to_datetime(dtree.time_coverage_start.item())
    if start_time.tzinfo is not None:
        start_time = start_time.tz_convert(None)

    for node in dtree.subtree:
        ds = node.to_dataset(inherit=False)
        ds[append_dim] = start_time
        ds[append_dim].attrs = {
            "description": "Volume Coverage Pattern time since start of volume scan"
        }
        ds = ds.set_coords(append_dim).expand_dims(dim=append_dim, axis=0)
        dtree[node.path].ds = ds

    return dtree


def ensure_dimension(dtree: DataTree, append_dim: str) -> DataTree:
    """
    Ensure that all datasets in a DataTree have a specified dimension.

    Parameters:
        dtree (DataTree): The DataTree to check and modify.
        append_dim (str): The name of the dimension to ensure.

    Returns:
        DataTree: The modified DataTree with the required dimension.
    """
    needs_expansion = not any(append_dim in node.dims for node in dtree.subtree)
    if needs_expansion:
        return exp_dim(dtree, append_dim)
    return dtree
