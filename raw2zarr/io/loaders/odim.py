import xradar as xd
from xarray import DataTree


def odim_loader(file: str) -> DataTree:
    return xd.io.open_odim_datatree(file)
