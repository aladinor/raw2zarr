import re

import icechunk
import pandas as pd
from xarray import Dataset, DataTree


def get_icechunk_repo(
    zarr_store: str,
) -> icechunk.Repository:
    storage = icechunk.local_filesystem_storage(zarr_store)
    try:
        return icechunk.Repository.create(storage)
    except icechunk.IcechunkError:
        return icechunk.Repository.open(storage)


def extract_timestamp(filename: str) -> pd.Timestamp:
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return pd.to_datetime(f"{date_part}{time_part}", format="%Y%m%d%H%M%S")

    match = re.search(r"[A-Z]{3}(\d{6})(\d{6})", filename)
    if match:
        date_part, time_part = match.groups()
        return pd.to_datetime(f"{date_part}{time_part}", format="%y%m%d%H%M%S")

    raise ValueError(f"Could not parse timestamp from filename: {filename}")


def remove_dims(dtree: DataTree, dim: str = "sweep") -> DataTree:
    def remove(ds: Dataset, dim: str = "sweep"):
        try:
            return ds.drop_dims(dim)
        except ValueError:
            return ds

    return dtree.map_over_datasets(remove, dim)
