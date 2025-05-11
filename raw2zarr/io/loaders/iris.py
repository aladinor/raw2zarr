from xarray import DataTree
from xradar.io import open_iris_datatree

from ..base import prepare2read


def iris_loader(file) -> DataTree:
    """
    Loads IRIS files from local, S3, or compressed sources using streaming.
    """
    stream = prepare2read(file)
    return open_iris_datatree(stream)
